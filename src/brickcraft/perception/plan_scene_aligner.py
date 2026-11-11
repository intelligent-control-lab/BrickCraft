"""Scene segmentation, reference alignment, outcome scoring, and state identification.

All mask comparisons share affine ECC, translation fallback, scale bounds
[0.5, 2.0], and IoU excluding enclosed reference holes. Identification segments
one frame once, then compares it with every plan state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

import cv2
import numpy as np

from brickcraft.utils.image_utils import resize_and_crop, structure_mask_iou


@dataclass
class StructureMatch:
    """Aligned-mask score and validity, shared by verification and identification."""

    iou: float
    scale: float
    rejected: bool = False
    reject_reason: str = ""


@dataclass
class StepIdentifierConfig:
    """Thresholds for :class:`PlanSceneAligner`.

    The highest-scoring candidate must fit at least ``min_iou`` to drive the arm.

    Candidates outside the fixed scale range [0.5, 2.0] are always rejected.
    """

    min_iou: float = 0.8


@dataclass
class StepScore:
    """How well one candidate state explains what the camera sees."""

    step_id: int          # 1..N: next step to build. N+1: completed-structure candidate.
    iou: float
    scale: float
    rejected: bool = False
    reject_reason: str = ""


@dataclass
class StepIdentification:
    """Result of one look at the bench."""

    step_id: Optional[int]      # next plan step to build; None when nothing fit
    is_complete: bool           # structure already matches the finished state
    iou: float
    confident: bool
    reason: str
    scores: List[StepScore] = field(default_factory=list)
    structure_mask: Optional[np.ndarray] = None

    def summary(self) -> str:
        where = "complete" if self.is_complete else f"step {self.step_id}"
        if self.step_id is None and not self.is_complete:
            where = "unknown"
        return (f"{where}  IoU {self.iou:.3f}  "
                f"{'confident' if self.confident else 'NOT confident: ' + self.reason}")


class PlanSceneAligner:

    def __init__(self, 
                 yolo_seg_structure_ckpt: str,
                 yolo_seg_brick_gripper_ckpt: str,
                 device: Optional[str] = None,
                 conf_threshold_structure: float = 0.5,
                 conf_threshold_brick_gripper: float = 0.5,
                 ):
        """
        Initialize YOLO model
        
        Args:
            yolo_seg_structure_ckpt: Path to structure segmentation model
            yolo_seg_brick_gripper_ckpt: Path to brick/gripper segmentation model
            conf_threshold_structure: Confidence threshold for structure segmentation (default: 0.5)
            conf_threshold_brick_gripper: Confidence threshold for brick/gripper segmentation (default: 0.5)
        """
        from ultralytics import YOLO
        import torch

        if not Path(yolo_seg_structure_ckpt).exists():
            raise FileNotFoundError(f"YOLO model weights not found: {yolo_seg_structure_ckpt}")
        if not Path(yolo_seg_brick_gripper_ckpt).exists():
            raise FileNotFoundError(f"YOLO model weights not found: {yolo_seg_brick_gripper_ckpt}")
        
        self.model_seg_structure = YOLO(yolo_seg_structure_ckpt)
        self.model_seg_brick_gripper = YOLO(yolo_seg_brick_gripper_ckpt)
        self.conf_threshold_structure = conf_threshold_structure
        self.conf_threshold_brick_gripper = conf_threshold_brick_gripper

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    
    def _seg_structure(self, image_rgb: np.ndarray, *, require_confident: bool = False) -> np.ndarray:
        """
        Args:
            image_rgb (np.ndarray): [H, W, 3] RGB image
        Returns:
            mask (np.ndarray): [H, W] binary mask of the segmented object
        """
        results = self.model_seg_structure.predict(
            source=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            task="segment",
            device=self.device,
            show=False,
            save=False,
            verbose=False,
        )
        
        if not results:
            raise ValueError("YOLO segmentation returned empty results.")
        
        if results[0].masks is None:
            raise ValueError("YOLO segmentation failed to produce masks. Model detected objects but no segmentation masks were generated.")
        
        conf = results[0].boxes.conf
        if len(conf) == 0:
            raise ValueError("YOLO segmentation returned no structure detections.")
        best_idx = int(conf.argmax().item())
        best_conf = float(conf[best_idx].item())
        
        # Check confidence threshold
        if not np.isfinite(best_conf):
            raise ValueError("YOLO structure confidence is not finite.")
        if best_conf < self.conf_threshold_structure:
            if require_confident:
                raise ValueError(
                    f"Structure confidence {best_conf:.3f} is below {self.conf_threshold_structure:.3f}"
                )
            print(f"Warning: Structure segmentation confidence ({best_conf:.3f}) is below threshold ({self.conf_threshold_structure:.3f})")
        
        best_mask = results[0].masks.data[best_idx]  # shape: (h, w)
        
        mask = (best_mask.cpu().numpy() > 0.5).astype(np.uint8) * 255 # Convert to binary mask
        if mask.shape[0] != image_rgb.shape[0] or mask.shape[1] != image_rgb.shape[1]:
            print("Warning: YOLO mask size does not match input image size. Resizing mask.")
            mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        return mask

    def _seg_brick_gripper(self, image_rgb: np.ndarray) -> Dict[Any, np.ndarray]:
        """
        Segment the brick and gripper using a separate YOLO model.
        Args:
            image_rgb (np.ndarray): [H, W, 3] RGB image
        Returns:
            masks: Dictionary of masks for different classes  (brick, gripper)
        """
        results = self.model_seg_brick_gripper.predict(
            source=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),  # It seems that the model was trained on BGR images
            task="segment",
            device=self.device,
            show=False,
            save=False,
            verbose=False,
        )
        if not results:
            raise ValueError("YOLO segmentation returned empty results.")
        
        if results[0].masks is None:
            raise ValueError("YOLO segmentation failed to produce masks. Model detected objects but no segmentation masks were generated.")
        
        
        masks = {}
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        for i, (mask, cls, conf) in enumerate(zip(results[0].masks.data.cpu().numpy(), classes, confidences)):
            class_name = self.model_seg_brick_gripper.names[int(cls)]
            
            # Check confidence threshold
            if conf < self.conf_threshold_brick_gripper:
                print(f"Warning: {class_name} segmentation confidence ({conf:.3f}) is below threshold ({self.conf_threshold_brick_gripper:.3f})")
            
            # Convert to binary mask
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Resize mask to match input image size if needed
            if mask.shape[0] != image_rgb.shape[0] or mask.shape[1] != image_rgb.shape[1]:
                print(f"Warning: YOLO {class_name} mask size does not match input image size. Resizing mask.")
                binary_mask = cv2.resize(binary_mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            masks[class_name] = binary_mask
        
        if not masks:
            raise ValueError("No masks detected for brick/gripper segmentation. Please check the model and confidence thresholds.")
        
        return masks

    def _extract_masks(self, combined_mask_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract plan assembly mask and reference brick mask from combined mask.
        
        Args:
            combined_mask_rgb (np.ndarray): [H, W, 3] Combined plan mask image.
        Returns:
            assembly_mask (np.ndarray): [H, W] Binary mask of the assembly
            reference_mask (np.ndarray): [H, W] Binary mask of the reference brick
        """
        
        # Define thresholds
        black_threshold = 30
        white_threshold = 200
        
        # 1. Extract assembly_mask: all non-black regions (white + colored)
        # A pixel is non-black if any channel is above the black threshold
        is_non_black = np.any(combined_mask_rgb > black_threshold, axis=2)

        # 2. Extract reference_mask: colored regions (not black, not white)
        # A pixel is white if all channels are above white_threshold
        is_white = np.all(combined_mask_rgb > white_threshold, axis=2)
        
        # A pixel is black if all channels are below black_threshold
        is_black = np.all(combined_mask_rgb < black_threshold, axis=2)

        # Reference mask: non-black AND non-white regions
        is_colored = (~is_black) & (~is_white)

        assembly_mask = (is_non_black).astype(np.uint8) * 255
        reference_mask = (is_colored).astype(np.uint8) * 255
        
        return assembly_mask, reference_mask

    def _find_mask_center(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find the point farthest from the edge (center of maximum inscribed circle) in the mask.
        Uses distance transform to find the most central point.
        
        Args:
            mask (np.ndarray): [H, W] Binary mask where the reference brick is highlighted.
        Returns:
            center (Tuple[int, int]): (x, y) coordinates of the farthest point from edge, or None if mask is empty.
        """
        mask_bool = (mask > 127).astype(np.uint8)
        if not mask_bool.any():
            return None
        
        # Calculate distance transform (distance to nearest zero pixel/edge)
        dist_transform = cv2.distanceTransform(mask_bool, cv2.DIST_L2, 5)
        
        # Find the point with maximum distance (farthest from edge)
        _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
        
        # max_loc is (x, y), convert to int tuple
        return (int(max_loc[0]), int(max_loc[1]))

    def _find_mask_bbox_center(self, mask: np.ndarray) -> Tuple[float, float]:
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            return mask.shape[1] / 2, mask.shape[0] / 2  # Default to image center if no mask
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        return cx, cy

    def _estimate_affine_ecc(
        self, template: np.ndarray, target: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate affine transformation from template to target using ECC algorithm.
        Args:
            template (np.ndarray): [H, W] Grayscale image (plan mask)
            target (np.ndarray): [H, W] Grayscale image (real obs mask)
            ECC failure falls back to the bounding-box translation seed.
        Returns:
            warp_matrix (np.ndarray): 2x3 affine transformation matrix
        """

        template_float = template.astype(np.float32) / 255.0
        target_float = target.astype(np.float32) / 255.0
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # 1. Initialize translation by aligning bounding-box centers.
        template_bbox_center = self._find_mask_bbox_center(template)
        target_bbox_center = self._find_mask_bbox_center(target)

        translation = (np.array(target_bbox_center) - np.array(template_bbox_center)).astype(np.float32)
        warp_matrix[:, 2] = translation

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        try:
            _, warp_matrix = cv2.findTransformECC(
                template_float, target_float, warp_matrix, cv2.MOTION_AFFINE, criteria
            )
        except cv2.error as e:
            print(f"ECC alignment failed: {e}")
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            warp_matrix[:, 2] = translation  # Keep the initial translation

        return warp_matrix

    @staticmethod
    def _check_warp(warp: np.ndarray) -> StructureMatch:
        """Validate an affine transform; scale is sqrt(det(linear transform))."""
        if not np.isfinite(warp).all():
            return StructureMatch(0.0, 0.0, True, "Invalid structure alignment")
        determinant = float(np.linalg.det(warp[:, :2].astype(np.float64)))
        if determinant <= 0:
            return StructureMatch(0.0, 0.0, True, "Invalid structure alignment")
        scale = float(np.sqrt(determinant))
        if not 0.5 <= scale <= 2.0:
            return StructureMatch(0.0, scale, True, f"scale {scale:.3f} outside [0.500, 2.000]")
        return StructureMatch(0.0, scale)

    def _match_structure_masks(self, expected: np.ndarray, observed: np.ndarray) -> StructureMatch:
        """Align two binary masks and score them with the shared validity rules."""
        if (expected.ndim != 2 or observed.ndim != 2 or expected.shape != observed.shape
                or not np.isfinite(expected).all() or not np.isfinite(observed).all()
                or not np.any(expected > 127) or not np.any(observed > 127)):
            return StructureMatch(0.0, 0.0, True, "Nonempty finite matching 2D masks are required")
        warp = self._estimate_affine_ecc(expected, observed)
        match = self._check_warp(warp)
        if match.rejected:
            return match
        aligned = cv2.warpAffine(
            expected, warp, (observed.shape[1], observed.shape[0]),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return StructureMatch(structure_mask_iou(aligned, observed), match.scale)

    def compute_structure_iou(self, scene_image_rgb: np.ndarray, expected_mask: np.ndarray) -> float:
        """Segment a new observation and compare it with the aligned goal silhouette.

        Reject missing/empty masks, weak detections, and invalid alignment scales.
        ECC failure uses bounding-box translation before scoring.
        Enclosed reference holes are excluded from the final IoU calculation.
        """
        if (not isinstance(scene_image_rgb, np.ndarray) or scene_image_rgb.ndim != 3
                or scene_image_rgb.shape[2] != 3 or scene_image_rgb.size == 0):
            raise ValueError("A nonempty RGB camera image is required for assembly verification")
        if (not isinstance(expected_mask, np.ndarray) or expected_mask.ndim != 2
                or expected_mask.size == 0 or not np.isfinite(expected_mask).all()):
            raise ValueError("A finite 2D goal_mask is required for assembly verification")

        observed = self._seg_structure(scene_image_rgb, require_confident=True)
        if (observed.ndim != 2 or observed.shape != scene_image_rgb.shape[:2]
                or not np.isfinite(observed).all()):
            raise ValueError("Observed structure mask must match the camera image")
        observed = (observed > 127).astype(np.uint8) * 255
        expected = self._resize_structure_mask(expected_mask, observed.shape)
        if not observed.any() or not expected.any():
            raise ValueError("Assembly verification requires nonempty observed and goal masks")

        match = self._match_structure_masks(expected, observed)
        if match.rejected:
            raise ValueError(match.reject_reason)
        return match.iou

    def align_and_generate_prompts(
        self,
        scene_image_rgb: np.ndarray,
        combined_mask_image_rgb: np.ndarray,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Align the scene image with the plan mask and generate prompts.
 
        Args:
            scene_image_rgb (np.ndarray): [H,W,3] The current scene image from the robot's camera.
            combined_mask_image_rgb (np.ndarray): [H,W,3] Combined plan mask for the current assembly step (black background, white assembly, and highlighted reference brick)

        Returns:
            Dictionary mapping obj_id to (points_array, labels_array) tuples.
                points_array: Nx2 array of [x, y] coordinates
                labels_array: N array of labels (1=positive, 0=negative)
            
            The prompts inlude reference brick, new brick and finger 3 objects.
        """
        
        # 1. Run YOLO segmentation to get the mask of the current image
        structure_mask = self._seg_structure(scene_image_rgb)
        brick_gripper_masks = self._seg_brick_gripper(scene_image_rgb)

        # 2. Resize the plan mask to match the structure mask size
        combined_mask_image_rgb = resize_and_crop(
            combined_mask_image_rgb,
            target_w=structure_mask.shape[1],
            target_h=structure_mask.shape[0],
            crop_mode="center",
            keep_aspect=True,
            scale_factor=1.0,
        )

        assembly_mask, reference_mask = self._extract_masks(combined_mask_image_rgb)

        # 3. Find the center point of the highlighted reference brick in the plan mask
        mask_center = self._find_mask_center(reference_mask)

        # 4. Estimate the best affine transform (ECC) from plan mask -> YOLO mask.
        warp_matrix = self._estimate_affine_ecc(assembly_mask, structure_mask)

        match = self._check_warp(warp_matrix)
        if match.rejected:
            raise ValueError(match.reject_reason)

        # 5. Apply the transform to the reference-brick center point.
        aligned_mask_center = cv2.transform(np.array([[mask_center]]), warp_matrix)[0][0].astype(int)

        # 6. Generate prompts based on the aligned mask and center point.
        prompt = {0: (aligned_mask_center.reshape(1, 2), np.array([1]))}  # Positive prompt at the center of the reference brick

        # Add prompts for brick and gripper masks
        for i, (class_name, mask) in enumerate(brick_gripper_masks.items(), start=1):
            mask_center = self._find_mask_center(mask)
            if mask_center is not None:
                prompt[i] = (np.array(mask_center).reshape(1, 2), np.array([1]))  # Positive prompt for each brick/gripper
            else:
                raise ValueError(f"Failed to find center for {class_name} mask. Please check the segmentation results and confidence thresholds.")
        return prompt

    def configure_plan(self, plan, config: Optional[StepIdentifierConfig] = None) -> None:
        """Set candidate states; reconfiguration clears resized template caches."""
        self.plan = plan
        self.identification_config = config or StepIdentifierConfig()
        self.templates = self._build_templates()
        self.complete_step_id = plan.num_steps + 1
        self._resized_cache: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {}

    def _build_templates(self) -> List[Tuple[int, np.ndarray]]:
        """One binary silhouette per state the bench can legitimately be in.

        Steps 1..N come from the authored ``current_mask`` (its non-black
        region is the whole structure; the coloured part is only the reference
        brick highlight). State N+1 is the final step's binary goal mask,
        rendered directly without a reference highlight.
        """
        templates: List[Tuple[int, np.ndarray]] = []

        for step in self.plan.steps:
            mask_rgb = step.current_mask
            is_structure = np.any(mask_rgb > 30, axis=2)   # non-black: white body + coloured highlight
            templates.append((step.step_id, is_structure.astype(np.uint8) * 255))

        templates.append((self.plan.num_steps + 1, self.plan.steps[-1].goal_mask))

        return templates

    @staticmethod
    def _resize_structure_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        resized = resize_and_crop(
            mask, target_h=shape[0], target_w=shape[1],
            crop_mode="center", keep_aspect=True, scale_factor=1.0,
        )
        return (resized > 127).astype(np.uint8) * 255

    def _templates_for(self, shape: Tuple[int, int]) -> Dict[int, np.ndarray]:
        """Binary templates resized to the camera frame, cached per frame size."""
        if shape not in self._resized_cache:
            self._resized_cache[shape] = {
                step_id: self._resize_structure_mask(template, shape)
                for step_id, template in self.templates
            }
        return self._resized_cache[shape]

    def _score_candidate(self, template: np.ndarray, structure_mask: np.ndarray,
                         step_id: int) -> StepScore:
        match = self._match_structure_masks(template, structure_mask)
        return StepScore(step_id, match.iou, match.scale, match.rejected, match.reject_reason)

    def identify(self, scene_rgb: np.ndarray) -> StepIdentification:
        """Identify which plan step the bench is at.

        Args:
            scene_rgb (np.ndarray): [H, W, 3] RGB frame from the fixed camera.
                Take it with the arm parked clear of the structure.
        Returns:
            :class:`StepIdentification`. ``step_id`` is the next step to build,
            ``is_complete`` says the structure is already finished, and
            ``confident`` is False when no candidate survives the scale limits
            or its IoU is too low; callers must not move on an unconfident reading.
        """
        if not hasattr(self, "plan"):
            raise RuntimeError("Call configure_plan before identifying a state")
        structure_mask = self._seg_structure(scene_rgb)
        templates = self._templates_for(structure_mask.shape[:2])

        scores = [self._score_candidate(template, structure_mask, step_id)
                  for step_id, template in templates.items()]

        accepted = sorted((s for s in scores if not s.rejected),
                          key=lambda s: s.iou, reverse=True)
        if not accepted:
            return StepIdentification(
                step_id=None, is_complete=False, iou=0.0, confident=False,
                reason="every candidate was rejected (invalid masks, alignment, or scale)",
                scores=scores, structure_mask=structure_mask,
            )

        best = accepted[0]
        is_complete = best.step_id == self.complete_step_id

        confident, reason = True, ""
        if best.iou < self.identification_config.min_iou:
            confident = False
            reason = f"best IoU {best.iou:.3f} < min_iou {self.identification_config.min_iou:.3f}"

        return StepIdentification(
            step_id=None if is_complete else best.step_id,
            is_complete=is_complete,
            iou=best.iou,
            confident=confident,
            reason=reason,
            scores=sorted(scores, key=lambda s: s.step_id),
            structure_mask=structure_mask,
        )
