import yaml
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any
from brickcraft.skills import AssemblySkillConfig, PickSkillConfig
from dataclasses import dataclass
from brickcraft.perception.plan_scene_aligner import StepIdentifierConfig

@dataclass
class SAM2Config():
    sam2_config: str
    sam2_checkpoint: str
    sam2_device: str = "cuda"


class LongHorizonPipelineConfig(BaseModel):
    """Robot, skill, perception, and execution-loop settings for assembly."""

    model_config = ConfigDict(extra="forbid")

    plan_path: str
    sam2_config: Dict[str, Any]

    yolo_seg_structure_ckpt: str
    yolo_seg_brick_gripper_ckpt: str

    pick_skill_config: Dict[str, Any]
    assemble_skill_config: Dict[str, Any]
    enable_step_identification: bool = False
    identification_min_iou: float = 0.8

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, encoding='utf-8') as f:
            return cls.model_validate(yaml.safe_load(f))
    
    def get_sam2_config(self) -> SAM2Config:
        """Get SAM2 config as SAM2Config object."""
        return SAM2Config(**self.sam2_config)

    def get_pick_skill_config(self) -> PickSkillConfig:
        """Get pick skill config as PickSkillConfig object."""
        return PickSkillConfig(**self.pick_skill_config)

    def get_assemble_skill_config(self) -> AssemblySkillConfig:
        """Get assemble skill config as AssemblySkillConfig object."""
        return AssemblySkillConfig(**self.assemble_skill_config)

    def get_step_identifier_config(self) -> StepIdentifierConfig:
        return StepIdentifierConfig(min_iou=self.identification_min_iou)
