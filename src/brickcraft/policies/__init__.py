"""BrickCraft policy module with auto-registration of brick_diffusion into LeRobot."""

import lerobot.policies.factory as _factory
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline

from brickcraft.policies.brick_diffusion.modeling_brick_diffusion import BrickDiffusionPolicy
from brickcraft.policies.brick_diffusion.configuration_brick_diffusion import BrickDiffusionConfig
from brickcraft.policies.brick_diffusion.processor_brick_diffusion import (
    make_brick_diffusion_pre_post_processors,
)

# ---------------------------------------------------------------------------
# Monkey-patch: get_policy_class
# ---------------------------------------------------------------------------
_original_get_policy_class = _factory.get_policy_class


def _patched_get_policy_class(name: str) -> type[PreTrainedPolicy]:
    if name == "brick_diffusion":
        return BrickDiffusionPolicy
    return _original_get_policy_class(name)


_factory.get_policy_class = _patched_get_policy_class

# ---------------------------------------------------------------------------
# Monkey-patch: make_policy_config
# ---------------------------------------------------------------------------
_original_make_policy_config = _factory.make_policy_config


def _patched_make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "brick_diffusion":
        return BrickDiffusionConfig(**kwargs)
    return _original_make_policy_config(policy_type, **kwargs)


_factory.make_policy_config = _patched_make_policy_config

# ---------------------------------------------------------------------------
# Monkey-patch: make_pre_post_processors
# ---------------------------------------------------------------------------
_original_make_pre_post_processors = _factory.make_pre_post_processors


def _patched_make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    if isinstance(policy_cfg, BrickDiffusionConfig) and pretrained_path is None:
        return make_brick_diffusion_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )
    return _original_make_pre_post_processors(policy_cfg, pretrained_path, **kwargs)


_factory.make_pre_post_processors = _patched_make_pre_post_processors
