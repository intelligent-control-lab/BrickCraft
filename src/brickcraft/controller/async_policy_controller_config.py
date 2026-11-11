"""
Configuration class for AsyncPolicyController.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AsyncPolicyControllerConfig:
    """Configuration for AsyncPolicyController."""

    # Policy parameters
    policy_type: str = "brick_diffusion"
    pretrained_path: str = ""
    
    # Control parameters
    control_freq: Optional[int] = None  # If None, will use robot_config.fps
    action_queue_thres: int = 3
    device: str = "cuda"
    
    # Segmentation parameters
    enable_segmentation: bool = False
    confirm_segmentation: bool = True  # Review initialization; press R to select prompts again.
    segmentation_camera: Optional[str] = None
    # Defaults to segmentation_camera for backwards-compatible in-place
    # replacement. Set to e.g. "fixed_mask" to preserve the source camera.
    segmentation_output_key: Optional[str] = None
    # Defaults to robot.config.image_seg_params. Use "multicolor_mask" for a
    # categorical RGB mask whose colors follow the number of tracked objects.
    segmentation_visualization_mode: Optional[str] = None
    show_segmentation: bool = False
    save_segmentation_masks: bool = False
    segmentation_save_dir: str = "./outputs/segmentation_masks"

    # Other parameters
    diffusion_loss_warning_thres: float = 1e-3 # when loss > this value, warning will be printed (potentially out of distribution)
