import numpy as np
import cv2
from typing import Literal


def structure_mask_iou(expected: np.ndarray, observed: np.ndarray) -> float:
    """Compare aligned structure masks, excluding enclosed holes in the reference.

    YOLO may label the interior of a structure as foreground. Pixels inside
    enclosed reference holes are excluded from both intersection and union;
    they do not count as matching foreground. Open gaps and missing exterior
    parts still contribute to the score. Inputs are 2D masks in the same frame.
    """
    if expected.ndim != 2 or observed.ndim != 2 or expected.shape != observed.shape:
        raise ValueError("Structure masks must be 2D arrays with matching shapes")
    expected_fg = expected > 127
    observed_fg = observed > 127
    contours, _ = cv2.findContours(
        expected_fg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros(expected.shape, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    holes = (filled > 0) & ~expected_fg
    intersection = np.count_nonzero(expected_fg & observed_fg & ~holes)
    union = np.count_nonzero((expected_fg | observed_fg) & ~holes)
    return float(intersection / union) if union else 0.0


def resize_and_crop(
    image: np.ndarray,
    target_h: int,
    target_w: int,
    crop_mode: Literal["center", "left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right", "bottom_center_right"] = "center",
    scale_factor: float = 1.0,
    keep_aspect: bool = True,
) -> np.ndarray:
    """
    Resize and crop an image with flexible cropping modes (center/edges/corners)
    and zoom-in scale factor.
    Example:
    - Resize to 224x224 with top-left crop and 1.2x zoom-in:
        resized_cropped_img = resize_and_crop(img, 224, 224, crop_mode="top_left", scale_factor=1.2)
    Args:
        image (np.ndarray): Input image as a numpy array (H, W, C).
    """

    h, w = image.shape[:2]

    # ----- Step 1: compute scale -----
    if keep_aspect:
        scale = max(target_w / w, target_h / h) * scale_factor
        scale_x = scale
        scale_y = scale
        new_w = int(round(w * scale_x))
        new_h = int(round(h * scale_y))
    else:
        scale_x = (target_w / w) * scale_factor
        scale_y = (target_h / h) * scale_factor
        new_w = int(round(w * scale_x))
        new_h = int(round(h * scale_y))

    # ----- Step 2: compute crop offsets in resized-space -----
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2

    if crop_mode == "left":
        start_x = 0
    elif crop_mode == "right":
        start_x = new_w - target_w
    elif crop_mode == "top":
        start_y = 0
    elif crop_mode == "bottom":
        start_y = new_h - target_h
    elif crop_mode == "top_left":
        start_x = 0
        start_y = 0
    elif crop_mode == "top_right":
        start_x = new_w - target_w
        start_y = 0
    elif crop_mode == "bottom_left":
        start_x = 0
        start_y = new_h - target_h
    elif crop_mode == "bottom_right":
        start_x = new_w - target_w
        start_y = new_h - target_h
    elif crop_mode == "bottom_center_right": # hardcoded! donot modify!
        # Bottom-aligned crop, shifted right from center.
        offset_x = (new_w - target_w) // 6  # shift 1/6 of the available range to the right
        start_x = (new_w - target_w) // 2 + offset_x
        start_y = new_h - target_h
    elif crop_mode == "center":
        pass
    else:
        raise ValueError(f"Unknown crop_mode '{crop_mode}'")

    # clip (in case zoom pushes sizes weirdly)
    start_x = np.clip(start_x, 0, max(0, new_w - target_w))
    start_y = np.clip(start_y, 0, max(0, new_h - target_h))

    # ----- Step 3: affine sampling from original image (subpixel-accurate crop) -----
    m = np.array(
        [
            [1.0 / scale_x, 0.0, start_x / scale_x],
            [0.0, 1.0 / scale_y, start_y / scale_y],
        ],
        dtype=np.float32,
    )

    warped = cv2.warpAffine(
        image,
        m,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return warped
