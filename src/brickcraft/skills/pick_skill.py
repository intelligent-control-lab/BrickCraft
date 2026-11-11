import time
from .base_skill import BaseSkill
import numpy as np
from typing import Dict, Any, TYPE_CHECKING
from brickcraft.controller import KinovaPose9dController
from dataclasses import dataclass
from brickcraft.planner.brick import COLOR_CODES, SHAPE_CODES

if TYPE_CHECKING:
    from brickcraft.perception.live_segmenter import LiveSegmenter


@dataclass
class PickSkillConfig:
    pick_poses_by_shape_and_color: dict[str, dict[str, list[float]]]  # shape -> color -> [x, y, z, rx, ry, rz]
    pick_prepare_pose: list[float]  # [x, y, z, rx, ry, rz] in metres and degrees; required in YAML.
    control_frequency: float = 10.0  # Hz
    linear_velocity: float = 0.05    # m/s
    angular_velocity: float = 30.0   # deg/s
    gripper_close_wait: float = 1.0  # s to wait for the gripper to close before lifting

    def __post_init__(self):
        if not isinstance(self.pick_poses_by_shape_and_color, dict) or not self.pick_poses_by_shape_and_color:
            raise ValueError("pick_poses_by_shape_and_color must be a nonempty shape-to-color pose mapping")
        for shape, colors in self.pick_poses_by_shape_and_color.items():
            if shape not in SHAPE_CODES or not isinstance(colors, dict) or not colors:
                raise ValueError(f"Invalid pick_poses_by_shape_and_color shape or color mapping: {shape!r}")
            for color, pose in colors.items():
                if color not in COLOR_CODES:
                    raise ValueError(f"Unsupported pick_poses_by_shape_and_color color: {color!r}; use full color names")
                pose = np.asarray(pose, dtype=float)
                if pose.shape != (6,) or not np.isfinite(pose).all():
                    raise ValueError(f"pick_poses_by_shape_and_color[{shape!r}][{color!r}] must be a finite 6D pose")
        if not np.isfinite(self.control_frequency) or self.control_frequency <= 0:
            raise ValueError("control_frequency must be finite and positive")
        if not np.isfinite(self.gripper_close_wait) or self.gripper_close_wait < 0:
            raise ValueError("gripper_close_wait must be finite and nonnegative")


class PickSkill(BaseSkill):
    """
    Skill to pick bricks according to the assembly plan.
    """

    def __init__(self,
                 robot_interface,
                 sam2_interface: "LiveSegmenter",
                 skill_cfg: PickSkillConfig,
                 ):
        """
        Args:
        robot_interface: lerobot robot object
        sam2_interface: live segmenter object
        skill_cfg: skill configuration object
        """
        super().__init__()
        self.robot_interface = robot_interface

        self.sam2_interface = sam2_interface       

        self.motion_controller = KinovaPose9dController(
            kinova=self.robot_interface,
            control_frequency=skill_cfg.control_frequency,
            linear_velocity=skill_cfg.linear_velocity,
            angular_velocity=skill_cfg.angular_velocity,
        )  # for scripted motion control

        self.pick_prepare_pose = np.array(skill_cfg.pick_prepare_pose)
        self.pick_poses_by_shape_and_color = {
            shape: {color: np.array(pose, dtype=float) for color, pose in colors.items()}
            for shape, colors in skill_cfg.pick_poses_by_shape_and_color.items()
        }
        self.gripper_close_wait = skill_cfg.gripper_close_wait

    def _resolve_pick_pose(self, assembly_step) -> np.ndarray:
        """Look up the target shape and color; require an explicit calibrated pose."""
        shape = assembly_step.b_tgt.shape
        color = assembly_step.b_tgt.color
        try:
            return self.pick_poses_by_shape_and_color[shape][color].copy()
        except KeyError as exc:
            raise ValueError(f"No pick pose configured for shape={shape!r}, color={color!r}") from exc

    def pre_condition(self, skill_input: Dict[str, Any]) -> bool:
        """
        Check the step input and reach the configured pick preparation pose.

        1. Check skill_input, a skill_input should contain:
            - assembly_step: AssemblyStep object
        2. Move to pick preparation pose
        """
        # 1. Check skill_input
        required_keys = ["assembly_step"]
        for key in required_keys:
            if key not in skill_input:
                print(f"[{self.__class__.__name__}] Missing required key: {key}")
                return False

        try:
            self._resolve_pick_pose(skill_input["assembly_step"])
        except (AttributeError, ValueError) as exc:
            print(f"[{self.__class__.__name__}] Invalid pick target: {exc}")
            return False

        step_id = skill_input["assembly_step"].step_id
        task_encoding = skill_input["assembly_step"].task_encoding
        brick_color = skill_input["assembly_step"].b_tgt.color

        print(f"[{self.__class__.__name__}] Assembly step: {step_id} | Brick color: {brick_color} | Task encoding: {task_encoding}")

        # 2. Move to pick preparation pose
        print(f"[{self.__class__.__name__}] Moving to pick preparation pose: {self.pick_prepare_pose}")
        if not self.motion_controller.move_cartesian6d(self.pick_prepare_pose):
            print(f"[{self.__class__.__name__}] Failed to move to pick preparation pose")
            return False

        return True

    def execute(self, skill_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move to the pose selected by target shape and color, grasp, then lift.
        """
        pick_pose6d = self._resolve_pick_pose(skill_input["assembly_step"])
        if not self.motion_controller.move_cartesian6d(pick_pose6d):
            raise RuntimeError("Failed to reach the pick pose")
        if not self.motion_controller.close_gripper():
            raise RuntimeError("Failed to send the gripper close command")
        time.sleep(self.gripper_close_wait)  # wait for gripper to close
        if not self.motion_controller.move_cartesian6d(self.pick_prepare_pose):
            raise RuntimeError("Failed to retreat to the pick preparation pose")
        return {}

    def post_condition(self, skill_input: Dict[str, Any]) -> bool:
        """Check that measured gripper position is greater than 0.5 after lifting."""
        try:
            position = self.robot_interface.get_current_gripper_position()
        except (RuntimeError, OSError) as exc:
            print(f"[{self.__class__.__name__}] Failed to read gripper feedback: {exc}")
            return False
        closed = bool(position is not None and np.isfinite(position) and position > 0.5)
        return closed
