from .base_skill import BaseSkill
from typing import Dict, Any, Optional
from brickcraft.controller import AsyncPolicyControllerConfig, AsyncPolicyController, KeyboardStopper
from dataclasses import dataclass, field
import time
import math
import cv2


@dataclass
class AssemblySkillConfig:
    assembly_prepare_pose: list[float]  # [x, y, z, rx, ry, rz] in metres and degrees; required in YAML.
    async_policy_controller_config: Dict[str, Any] = field(default_factory=dict)
    episode_timeout: float = 60.0         # s, hard cap on one policy episode
    gripper_open_threshold: float = 0.4   # gripper_position below this counts as open
    open_wait_duration: float = 3.0       # Terminate the policy after the gripper stays open for this many seconds.
    post_condition_iou_threshold: float = 0.8  # Calibrate against observed assembly outcomes.

    def __post_init__(self):
        if (not math.isfinite(self.post_condition_iou_threshold)
                or not 0 < self.post_condition_iou_threshold < 1):
            raise ValueError("post_condition_iou_threshold must be finite and strictly between 0 and 1")

    def get_async_policy_controller_config(self) -> AsyncPolicyControllerConfig:
        """Get async policy controller config as AsyncPolicyControllerConfig object."""
        return AsyncPolicyControllerConfig(**self.async_policy_controller_config)


class AssemblySkill(BaseSkill):
    """
    Skill to assemble bricks according to the assembly plan.
    """

    def __init__(self,
                 robot_interface,
                 sam2_interface,
                 aligner_interface,
                 skill_cfg: AssemblySkillConfig,
                 ):
        super().__init__()
        self.robot_interface = robot_interface
        self.sam2_interface = sam2_interface
        self.aligner_interface = aligner_interface
        self.skill_cfg = skill_cfg
        self.async_policy_controller_config = skill_cfg.get_async_policy_controller_config()
        self.policy_controller = AsyncPolicyController(
            robot_interface = self.robot_interface,
            sam2_interface = self.sam2_interface,
            config = self.async_policy_controller_config
        )

    def pre_condition(self, skill_input: Dict[str, Any]) -> bool:
        """Check the required assembly_step input without operator confirmation."""
        required_keys = ["assembly_step"]
        for key in required_keys:
            if key not in skill_input:
                print(f"[{self.__class__.__name__}] Missing required key: {key}")
                return False

        step_id = skill_input["assembly_step"].step_id
        task_encoding = skill_input["assembly_step"].task_encoding
        brick_color = skill_input["assembly_step"].b_tgt.color

        print(f"[{self.__class__.__name__}] Assembly step: {step_id} | Brick color: {brick_color} | Task encoding: {task_encoding}")

        return True

    def execute(self, skill_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the brick assembly skill.
        1. Set task encoding
        2. plan-scene alignment
        3. run controller
        """
        # 1. Set task encoding
        task_encoding = skill_input["assembly_step"].task_encoding
        step_id = skill_input["assembly_step"].step_id
        if self.policy_controller.has_task_encoding:
            self.policy_controller.set_task(task_encoding)
        self.policy_controller.set_segmentation_save_context(step_id=step_id)

        # 2. Plan-scene alignment
        robot_obs = self.robot_interface.get_observation()
        scene_image = robot_obs["fixed_camera"]
        combined_mask = skill_input["assembly_step"].current_mask
        sam2_prompts = self.aligner_interface.align_and_generate_prompts(scene_image_rgb=scene_image, combined_mask_image_rgb=combined_mask)

        # 3. Run controller
        # Terminate the policy after the gripper stays open for open_wait_duration seconds.
        open_threshold = self.skill_cfg.gripper_open_threshold # gripper_position < open_threshold is considered as "open"
        wait_duration = self.skill_cfg.open_wait_duration
        open_start_time: Optional[float] = None
        last_is_open = False # To track the last state of the gripper (open or closed)
        def termination_condition(obs: Dict[str, Any]) -> bool:
            nonlocal open_start_time, last_is_open
            gripper_key = next((key for key in obs.keys() if "gripper_position" in key), None)
            if gripper_key is None:
                return False
            gripper_position = obs.get(gripper_key)
            if gripper_position is None:
                return False
            is_open = gripper_position < open_threshold
            if is_open and not last_is_open: 
                open_start_time = time.time()
            if not is_open: # reset the timer if the gripper is closed
                open_start_time = None
            last_is_open = is_open
            if is_open and open_start_time is not None:
                return (time.time() - open_start_time) >= wait_duration
            return False

        stopper = KeyboardStopper(self.policy_controller) # Listen for 'q' key to stop

        stopper.start()
        self.policy_controller.run_episode(
            episode_timeout=self.skill_cfg.episode_timeout,
            sam2_prompts=sam2_prompts,
            termination_condition=termination_condition,
        )
        stopper.stop()

        return {}

    def post_condition(self, skill_input: Dict[str, Any]) -> bool:
        """Verify the expected post-assembly structure using a new YOLO segmentation."""
        try:
            step = skill_input["assembly_step"]
            scene = self.robot_interface.get_observation()["fixed_camera"]
            iou = self.aligner_interface.compute_structure_iou(scene, step.goal_mask)
            if not math.isfinite(iou):
                raise ValueError("Assembly IoU is not finite")
        except (KeyError, AttributeError, ValueError, RuntimeError, OSError, cv2.error) as exc:
            print(f"[{self.__class__.__name__}] Assembly verification failed: {exc}")
            return False

        threshold = self.skill_cfg.post_condition_iou_threshold
        success = iou > threshold
        print(f"[{self.__class__.__name__}] Post-condition IoU: {iou:.3f}; "
              f"threshold: {threshold:.3f}; success: {success}")
        return success
