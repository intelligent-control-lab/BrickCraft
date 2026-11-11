import colorsys
from typing import Dict, List, Literal, Optional, Tuple, Union
import cv2
import numpy as np

TrackedMasks = Dict[Union[str, int], Optional[np.ndarray]]
VisualizationMode = Literal["background_dim", "bounding_box", "multicolor_mask"]
"""
Supported mask visualization modes:
    background_dim: Dim the background while keeping masked regions bright.
    bounding_box: Draw bounding boxes around object masks.
    multicolor_mask: Render one solid RGB color per object on a black background.

Example:
    result = MaskVisualizer.visualize(
        img_rgb, masks, mode="background_dim", dim_factor=0.3,
        add_contours=True, contour_thickness=2, expand_pixels=0, fade_distance=5,
    )
"""

class MaskVisualizer:
    """Handles different mask visualization modes."""
    
    # Color palette for different objects
    COLORS = [
        (0, 0, 255),      # Red
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (128, 0, 255),    # Purple
        (255, 128, 0),    # Orange
    ]

    # Keep the first three RGB colors stable for categorical policy inputs:
    # red, yellow, and cyan. Additional colors are generated deterministically.
    MULTICOLOR_MASK_BASE_COLORS = [
        (255, 0, 0),      # Red
        (255, 255, 0),    # Yellow
        (0, 255, 255),    # Cyan
    ]
    
    @staticmethod
    def get_color(idx: int) -> Tuple[int, int, int]:
        """Get a unique color for each object index."""
        return MaskVisualizer.COLORS[idx % len(MaskVisualizer.COLORS)]

    @staticmethod
    def get_multicolor_mask_colors(count: int) -> List[Tuple[int, int, int]]:
        """Return ``count`` deterministic RGB colors for semantic mask objects.

        The first three colors are always red, yellow, and cyan for backwards
        compatibility. A golden-ratio hue sequence supplies as many additional
        colors as the current prompt/mask count requires.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")

        colors = list(MaskVisualizer.MULTICOLOR_MASK_BASE_COLORS[:count])
        golden_ratio_conjugate = 0.618033988749895
        for extra_idx in range(max(0, count - len(colors))):
            hue = (0.08 + extra_idx * golden_ratio_conjugate) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
            colors.append(tuple(int(round(channel * 255)) for channel in rgb))
        return colors

    @staticmethod
    def multicolor_mask(
        masks: TrackedMasks,
        shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Render every tracked object as a distinct color on black.

        Object order is deterministic (sorted by object id), and the number of
        colors follows the number of prompts/masks instead of being capped at
        three. Later objects take precedence where masks overlap.
        """
        ordered_masks = sorted(masks.items(), key=lambda item: str(item[0]))
        if shape is None:
            first_mask = next((mask for _, mask in ordered_masks if mask is not None), None)
            if first_mask is None:
                raise ValueError("shape is required when masks contains no valid mask")
            first_mask_array = np.asarray(first_mask).squeeze()
            if first_mask_array.ndim != 2:
                raise ValueError(
                    f"Expected a 2D mask after squeeze, got shape {first_mask_array.shape}"
                )
            shape = first_mask_array.shape

        mask_rgb = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        colors = MaskVisualizer.get_multicolor_mask_colors(len(ordered_masks))
        for color, (obj_id, mask) in zip(colors, ordered_masks):
            if mask is None:
                continue
            mask_bool = np.asarray(mask).squeeze().astype(bool)
            if mask_bool.shape != shape:
                raise ValueError(
                    f"Mask for object {obj_id!r} has shape {mask_bool.shape}; expected {shape}"
                )
            mask_rgb[mask_bool] = np.asarray(color, dtype=np.uint8)
        return mask_rgb
    
    @staticmethod
    def visualize(
        frame: np.ndarray,
        masks: TrackedMasks,
        mode: VisualizationMode,
        **kwargs
    ) -> np.ndarray:
        """
        Unified interface for mask visualization with different modes.
        
        Args:
            frame: Input RGB frame
            masks: Dictionary of object masks
            mode: Visualization mode (background_dim, bounding_box, multicolor_mask)
            **kwargs: Mode-specific parameters
        
        Returns:
            Visualized frame
        
        """
        if mode == "background_dim":
            return MaskVisualizer.background_dim(frame, masks, **kwargs)

        elif mode == "bounding_box":
            return MaskVisualizer.bounding_box(frame, masks, **kwargs)

        elif mode == "multicolor_mask":
            return MaskVisualizer.multicolor_mask(masks, shape=frame.shape[:2], **kwargs)
        
        else:
            print(f"Warning: Unknown visualization mode '{mode}', using original image")
            return frame
    
    @staticmethod
    def expand_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
        """
        Expand a binary mask outward by a given number of pixels using morphological dilation.
        
        Args:
            mask: Binary mask (bool or uint8)
            pixels: Number of pixels to expand outward (0 = no expansion)
        
        Returns:
            Expanded binary mask (same dtype as input)
        """
        if pixels <= 0:
            return mask
        
        original_dtype = mask.dtype
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Create circular kernel for uniform expansion
        kernel_size = 2 * pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply dilation
        expanded = cv2.dilate(mask_uint8, kernel, iterations=1)
        
        if original_dtype == bool:
            return expanded > 0
        return (expanded > 0).astype(original_dtype)
    


    @staticmethod
    def background_dim(
        frame: np.ndarray,
        masks: TrackedMasks,
        dim_factor: float = 0.3,
        add_contours: bool = True,
        contour_thickness: int = 2,
        expand_pixels: int = 0,
        fade_distance: int = 0
    ) -> np.ndarray:
        """
        Dim the background while keeping masked regions bright.
        
        Args:
            frame: Input BGR frame
            masks: Dictionary of object masks
            dim_factor: Brightness multiplier for background (0.0-1.0)
            add_contours: Whether to add contour lines around objects
            contour_thickness: Thickness of contour lines
            expand_pixels: Number of pixels to expand masks outward (0 = no expansion)
            fade_distance: Distance in pixels for gradual fade from mask edge to background (0 = sharp transition)
        """
        result = frame.copy().astype(np.float32)
        
        # Create combined mask for all objects (with expansion)
        combined_mask = np.zeros(frame.shape[:2], dtype=bool)
        expanded_masks = {}
        for obj_id, mask in masks.items():
            if mask is not None:
                exp_mask = MaskVisualizer.expand_mask(mask, expand_pixels) if expand_pixels > 0 else mask
                expanded_masks[obj_id] = exp_mask
                combined_mask |= exp_mask
        
        if fade_distance > 0:
            # Create a distance transform to get gradual fade
            # Distance from each pixel to the nearest masked pixel
            combined_mask_uint8 = combined_mask.astype(np.uint8) * 255
            dist_transform = cv2.distanceTransform(
                cv2.bitwise_not(combined_mask_uint8), 
                cv2.DIST_L2, 
                5
            )
            
            # Create fade mask: 1.0 at object edges, gradually decreasing to dim_factor
            # Pixels within fade_distance pixels from mask edge will have values between 1.0 and dim_factor
            fade_mask = np.ones_like(dist_transform, dtype=np.float32)
            fade_region = (dist_transform > 0) & (dist_transform <= fade_distance)
            
            # Linear interpolation: closer to mask = brighter (closer to 1.0)
            # farther from mask = dimmer (closer to dim_factor)
            fade_mask[fade_region] = 1.0 - (dist_transform[fade_region] / fade_distance) * (1.0 - dim_factor)
            fade_mask[dist_transform > fade_distance] = dim_factor
            fade_mask[combined_mask] = 1.0  # Keep masked regions at full brightness
            
            # Apply the fade mask to each channel
            for c in range(3):
                result[:, :, c] *= fade_mask
        else:
            # Sharp transition (original behavior)
            result[~combined_mask] *= dim_factor
        
        # Clip values to valid range [0, 255]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Optional: add contours
        if add_contours:
            for idx, (obj_id, mask) in enumerate(
                sorted(expanded_masks.items(), key=lambda item: str(item[0]))
            ):
                if mask is None:
                    continue
                
                color = MaskVisualizer.get_color(idx)
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(result, contours, -1, color, contour_thickness)
        
        return result

    @staticmethod
    def bounding_box(
        frame: np.ndarray,
        masks: TrackedMasks,
        thickness: int = 2,
        expand_pixels: int = 0,
        box_color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes around each mask.
        
        Args:
            frame: Input RGB frame
            masks: Dictionary of object masks
            thickness: Thickness of the bounding box lines
            expand_pixels: Number of pixels to expand masks outward before calculating bounding box (0 = no expansion)
            box_color: Color of the bounding box in RGB format. If None, uses the default color palette.
        """
        result = frame.copy()
        
        for idx, (obj_id, mask) in enumerate(
            sorted(masks.items(), key=lambda item: str(item[0]))
        ):
            if mask is None:
                continue

            # Optionally expand mask before computing box.
            if expand_pixels > 0:
                mask = MaskVisualizer.expand_mask(mask, expand_pixels)

            mask_bool = mask.astype(bool)
            if not np.any(mask_bool):
                continue

            ys, xs = np.where(mask_bool)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())

            color = box_color if box_color is not None else MaskVisualizer.get_color(idx)
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, thickness)
        
        return result
