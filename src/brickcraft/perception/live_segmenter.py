"""
Live Video Segmentation with SAM2 Camera Predictor.

ObjectSelector: 
   - Interactive object selection
   - Support for multiple object tracking simultaneously
   - Positive and negative sample selection
   - Multi-object selection in a single session

Usage Example:
--------------
    from sam2.build_sam import build_sam2_camera_predictor
    from brickcraft.perception.live_segmenter import LiveSegmenter, ObjectSelector
    
    # Build SAM2 predictor
    predictor = build_sam2_camera_predictor(
        config_file="sam2.1_hiera_s",
        ckpt_path="checkpoint.pt"
    )
    
    # Initialize segmenter
    segmenter = LiveSegmenter(predictor)
    
    # First frame: interactive selection
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    mask_logits, masks = segmenter.initialize_tracking(first_frame_rgb)
    # User interactions:
    #   - Left click: positive samples (+)
    #   - Right click: negative samples (-)
    #   - Press 'N': create new object
    #   - Press Enter: finish selection
    
    # Subsequent frames: automatic tracking
    for frame in video_frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask_logits, masks = segmenter.track_frame(frame_rgb)
        # Use masks for visualization or further processing

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np


class ObjectSelector:
    """
    Interactive object selector with support for multiple objects and positive/negative samples.
    
    Features:
    - Left click: Add positive sample (+)
    - Right click: Add negative sample (-)
    - Press 'N' or click button: Create new object
    - Press Enter: Finish selection
    - Each object requires at least one positive sample
    
    Example:
        selector = ObjectSelector()
        objects_data = selector.select_objects(frame_bgr)
        # Returns: {obj_id: (points_array, labels_array), ...}
    """
    
    def __init__(self):
        self.colors = ['green', 'blue', 'yellow', 'cyan', 'magenta', 'orange']
    
    def select_objects(self, frame_bgr: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Interactive multi-object selection on a frame.
        
        Args:
            frame_bgr: BGR image frame
            
        Returns:
            Dictionary mapping object_index to (points_array, labels_array) tuples.
        """
        objects_clicks = self._select_multiple_points_gui(frame_bgr)
        if not objects_clicks:
            objects_clicks = self._select_multiple_points_manual(frame_bgr)
        if not objects_clicks:
            return {}
        
        # Convert to numpy arrays
        result: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for obj_index, clicks in objects_clicks.items():
            positive_points = [(x, y) for x, y, label in clicks if label == 1]
            negative_points = [(x, y) for x, y, label in clicks if label == 0]
            
            if not positive_points:
                print(f"Warning: Object {obj_index} has no positive samples. Skipping.")
                continue
            
            all_points = positive_points + negative_points
            all_labels = [1] * len(positive_points) + [0] * len(negative_points)
            
            points_array = np.array(all_points, dtype=np.float32)
            labels_array = np.array(all_labels, dtype=np.int32)
            
            result[obj_index] = (points_array, labels_array)
            print(f"✓ Object {obj_index}: {len(positive_points)} positive, {len(negative_points)} negative samples")
        
        return result
    
    def _select_multiple_points_gui(self, frame_bgr: np.ndarray) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        GUI-based interactive selection using matplotlib.
        
        Returns:
            Dictionary mapping object_index to list of (x, y, label) tuples.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Button
        except ImportError:
            print("matplotlib not available for GUI selection.")
            return {}

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15)
        ax.imshow(rgb)
        ax.axis("off")
        
        # State
        current_obj_id = [1]  # Use list to make it mutable in nested functions
        objects_points: Dict[int, List[Tuple[int, int, int]]] = {1: []}
        
        def update_title():
            pos_count = sum(1 for _, _, label in objects_points[current_obj_id[0]] if label == 1)
            neg_count = sum(1 for _, _, label in objects_points[current_obj_id[0]] if label == 0)
            ax.set_title(
                f"Object {current_obj_id[0]} | Positive: {pos_count}, Negative: {neg_count}\n"
                f"Left click: + | Right click: - | 'N': New Object | Enter: Done",
                fontsize=10
            )
            fig.canvas.draw_idle()

        def _on_click(event):
            if event.inaxes != ax or event.xdata is None:
                return
            x, y = int(event.xdata), int(event.ydata)
            
            obj_id = current_obj_id[0]
            color = self.colors[(obj_id - 1) % len(self.colors)]
            
            if event.button == 1:  # Left click - positive sample
                label = 1
                objects_points[obj_id].append((x, y, label))
                ax.scatter([x], [y], c=color, s=80, marker="+", linewidths=3, edgecolors='black')
            elif event.button == 3:  # Right click - negative sample
                label = 0
                objects_points[obj_id].append((x, y, label))
                ax.scatter([x], [y], c=color, s=80, marker="x", linewidths=3, edgecolors='black')
            else:
                return
            
            update_title()

        def _on_key(event):
            if event.key in {"enter", "return"}:
                # Check if current object has at least one positive sample
                if current_obj_id[0] in objects_points:
                    pos_count = sum(1 for _, _, label in objects_points[current_obj_id[0]] if label == 1)
                    if pos_count == 0 and len(objects_points[current_obj_id[0]]) > 0:
                        print(f"Warning: Object {current_obj_id[0]} has no positive samples. It will be ignored.")
                        del objects_points[current_obj_id[0]]
                plt.close(fig)
            elif event.key in {"escape", "q"}:
                objects_points.clear()
                plt.close(fig)
            elif event.key in {"n", "N"}:  # New object
                obj_id = current_obj_id[0]
                if obj_id in objects_points:
                    pos_count = sum(1 for _, _, label in objects_points[obj_id] if label == 1)
                    if pos_count == 0:
                        print(f"Cannot create new object: Current object {obj_id} needs at least one positive sample!")
                        return
                    elif len(objects_points[obj_id]) == 0:
                        return
                
                # Create new object
                current_obj_id[0] += 1
                objects_points[current_obj_id[0]] = []
                print(f"Switched to Object {current_obj_id[0]}")
                update_title()

        # Add button for new object
        ax_button = plt.axes((0.4, 0.02, 0.2, 0.05))
        btn_new_obj = Button(ax_button, 'New Object (N)')
        
        def on_button_click(event):
            class KeyEvent:
                def __init__(self):
                    self.key = 'n'
            _on_key(KeyEvent())
        
        btn_new_obj.on_clicked(on_button_click)

        fig.canvas.mpl_connect("button_press_event", _on_click)
        fig.canvas.mpl_connect("key_press_event", _on_key)
        
        update_title()
        plt.show()
        
        # Filter out objects without positive samples
        filtered_objects = {}
        for obj_id, points in objects_points.items():
            pos_count = sum(1 for _, _, label in points if label == 1)
            if pos_count > 0:
                filtered_objects[obj_id] = points
            elif len(points) > 0:
                print(f"Object {obj_id} ignored: no positive samples")
        
        return filtered_objects
    
    def _select_multiple_points_manual(self, frame_bgr: np.ndarray) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Manual text-based selection (fallback when GUI is unavailable).
        
        Returns:
            Dictionary mapping object_index to list of (x, y, label) tuples.
        """
        fallback_path = Path("select_click_frame.png")
        cv2.imwrite(str(fallback_path), frame_bgr)
        print(
            "GUI backends unavailable; saved frame to select_click_frame.png. "
            "Enter coords as 'x y label' per line (label: 1=positive, 0=negative).\n"
            "Type 'next' to start a new object, blank to finish all objects."
        )
        
        objects_points: Dict[int, List[Tuple[int, int, int]]] = {}
        current_obj_id = 1
        objects_points[current_obj_id] = []
        
        while True:
            coords = input(f"Object {current_obj_id} point (or 'next'/'done'): ").strip().lower()
            
            if coords in {'done', ''}:
                if current_obj_id in objects_points:
                    pos_count = sum(1 for _, _, label in objects_points[current_obj_id] if label == 1)
                    if pos_count == 0 and len(objects_points[current_obj_id]) > 0:
                        print(f"Warning: Object {current_obj_id} has no positive samples, removing it.")
                        del objects_points[current_obj_id]
                break
            
            if coords == 'next':
                pos_count = sum(1 for _, _, label in objects_points[current_obj_id] if label == 1)
                if pos_count == 0:
                    print(f"Cannot create new object: Current object {current_obj_id} needs at least one positive sample!")
                    continue
                
                current_obj_id += 1
                objects_points[current_obj_id] = []
                print(f"Switched to Object {current_obj_id}")
                continue
            
            try:
                parts = coords.split()
                if len(parts) == 2:
                    x_str, y_str = parts
                    label = 1
                elif len(parts) == 3:
                    x_str, y_str, label_str = parts
                    label = int(label_str)
                    if label not in {0, 1}:
                        print("Label must be 0 or 1. Try again.")
                        continue
                else:
                    print("Invalid input, expected 'x y' or 'x y label'. Try again.")
                    continue
                
                objects_points[current_obj_id].append((int(x_str), int(y_str), label))
            except ValueError:
                print("Invalid input format. Try again.")
        
        return objects_points


class LiveSegmenter:
    """Wrap SAM2 predictors for interactive tracking in camera mode only."""

    def __init__(
        self,
        predictor,
        obj_id: int = 1,
    ):
        self.predictor = predictor
        self.obj_points: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # obj_id -> (points_array, labels_array)
        self.default_obj_id = obj_id
        self.initialized = False
        self.selector = ObjectSelector() # interactive object selector


    def _collect_prompts(
        self, frame_bgr: np.ndarray
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Collect multiple seed clicks from the user for multiple objects.
        
        Args:
            frame_bgr: BGR image frame
            
        Returns:
            Dictionary mapping object_id to (points_array, labels_array) tuples.
            points_array: Nx2 array of [x, y] coordinates
            labels_array: N array of labels (1=positive, 0=negative)
        """
        objects_data = self.selector.select_objects(frame_bgr)
        
        if not objects_data:
            return {}

        result: Dict[Union[str, int], Tuple[np.ndarray, np.ndarray]] = {}
        
        for obj_index, (points_array, labels_array) in objects_data.items():
            # Determine object name
            if self.default_obj_id is not None and obj_index == 1:
                obj_name = self.default_obj_id
            else:
                obj_name = f"object_{obj_index}"
            
            result[obj_name] = (points_array, labels_array)
        
        return result


    def initialize_tracking(self, frame_rgb: np.ndarray):
        """
        Using interactive object selection to initialize tracking on the first frame.
        """
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.obj_points = self._collect_prompts(frame_bgr)
        if not self.obj_points:
            print("No objects were selected; aborting initialization.")
            return None

        if not hasattr(self.predictor, "load_first_frame"):
            raise RuntimeError("Camera mode requires a predictor with load_first_frame.")
        self.predictor.load_first_frame(frame_rgb)

        out_obj_ids_all: list[Union[str, int]] = []
        out_mask_logits_all: list[np.ndarray] = []
        for obj_id, (points, labels) in self.obj_points.items():
            if points is None or len(points) == 0:
                continue
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0, obj_id=obj_id, points=points, labels=labels
            )
            out_obj_ids_all.extend(out_obj_ids)
            out_mask_logits_all.extend(out_mask_logits)
        if not out_obj_ids_all:
            print("No prompts were registered; aborting initialization.")
            return None
        mask_logits = self._mask_logits_by_obj(out_obj_ids_all, out_mask_logits_all)
        masks = {
            obj_id: self._mask_from_logits(logits)
            for obj_id, logits in mask_logits.items()
        }

        self.initialized = True
        return mask_logits, masks


    def initialize_tracking_with_prompts(self, frame_rgb: np.ndarray, obj_points: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """
        Initialize tracking with given prompts
        Args:
            frame_rgb: HxWx3 numpy array in RGB format.
            obj_points: dict of object_id to (points_array, labels_array) tuples.
                points_array: Nx2 array of [x, y] coordinates
                labels_array: N array of labels (1=positive, 0=negative)

        """
        self.obj_points = obj_points

        if not hasattr(self.predictor, "load_first_frame"):
            raise RuntimeError("Camera mode requires a predictor with load_first_frame.")
        self.predictor.load_first_frame(frame_rgb)

        out_obj_ids_all: list[Union[str, int]] = []
        out_mask_logits_all: list[np.ndarray] = []
        for obj_id, (points, labels) in self.obj_points.items():
            if points is None or len(points) == 0:
                continue
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0, obj_id=obj_id, points=points, labels=labels
            )
            out_obj_ids_all.extend(out_obj_ids)
            out_mask_logits_all.extend(out_mask_logits)
        if not out_obj_ids_all:
            print("No prompts were registered; aborting initialization.")
            return None
        mask_logits = self._mask_logits_by_obj(out_obj_ids_all, out_mask_logits_all)
        masks = {
            obj_id: self._mask_from_logits(logits)
            for obj_id, logits in mask_logits.items()
        }

        self.initialized = True
        return mask_logits, masks



    def track_frame(self, frame_rgb: np.ndarray):
        """
        Propagate masks to the next frame.
        Args:
            frame_rgb: HxWx3 numpy array in RGB format.
        Returns:
            mask_logits: dict of object id to mask logits (HxW numpy arrays)
            masks: dict of object id to binary masks (HxW numpy arrays)
        """
        if not self.initialized:
            raise RuntimeError("initialize_tracking must be called before track_frame")

        if not hasattr(self.predictor, "track"):
            raise RuntimeError("Camera mode requires predictor.track.")
        out_obj_ids, out_mask_logits = self.predictor.track(frame_rgb)
        mask_logits = self._mask_logits_by_obj(out_obj_ids, out_mask_logits)
        masks = {
            obj_id: self._mask_from_logits(logits)
            for obj_id, logits in mask_logits.items()
        }

        return mask_logits, masks

    @staticmethod
    def _mask_logits_by_obj(out_obj_ids, out_mask_logits):
        """Return a dict of mask logits keyed by object id."""
        return {obj_id: out_mask_logits[idx] for idx, obj_id in enumerate(out_obj_ids)}

    @staticmethod
    def _mask_from_logits(mask_logits):
        if mask_logits is None:
            return None
        mask_tensor = (mask_logits > 0.0).squeeze(0)
        return mask_tensor.detach().cpu().numpy().astype(bool)

