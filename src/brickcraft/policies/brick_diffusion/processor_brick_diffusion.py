from typing import Any
import torch
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from .configuration_brick_diffusion import BrickDiffusionConfig

def make_brick_diffusion_pre_post_processors(
    config: BrickDiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for Brick Diffusion Policy.
    Uses LeRobot's diffusion pre/post-processors.
    """
    return make_diffusion_pre_post_processors(config, dataset_stats)
