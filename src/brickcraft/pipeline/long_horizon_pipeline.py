"""Identification: low IoU -> wait 1 s -> re-identify; complete -> exit.
Otherwise: pick -> assembly -> re-identify, even if the step is unchanged.
Sequential: run steps 1..N -> exit.
Skill failure: y -> re-identify (identification) / skip step (sequential); n -> exit.
Always clean up on exit.
"""

from __future__ import annotations

import time

import cv2
from sam2.build_sam import build_sam2_camera_predictor

from brickcraft.skills import PickSkill, AssemblySkill
from brickcraft.robots.Kinova import KinovaPose9D, KinovaPose9DConfig
from brickcraft.perception.live_segmenter import LiveSegmenter
from brickcraft.perception.plan_scene_aligner import PlanSceneAligner, StepIdentification

from .assembly_plan import AssemblyPlan
from .long_horizon_pipeline_config import LongHorizonPipelineConfig


class LongHorizonPipeline:
    """Select, execute, and verify progress through the assembly plan."""

    def __init__(self, config: LongHorizonPipelineConfig):
        self.cfg = config

        self.plan = AssemblyPlan(self.cfg.plan_path)
        robot_config = KinovaPose9DConfig()
        self.robot_interface = KinovaPose9D(robot_config)
        self.robot_interface.connect()
        print(f"Robot {self.robot_interface.config.type} connected.")

        # Initialize SAM2 segmenter
        sam2_cfg = self.cfg.get_sam2_config()
        sam2_predictor = build_sam2_camera_predictor(
            config_file=sam2_cfg.sam2_config,
            ckpt_path=sam2_cfg.sam2_checkpoint,
            device=sam2_cfg.sam2_device
        )
        self.sam2_interface = LiveSegmenter(predictor=sam2_predictor)
        print("SAM2 segmenter initialized.")

        # Initialize plan scene aligner
        self.scene_aligner = PlanSceneAligner(
            yolo_seg_structure_ckpt=self.cfg.yolo_seg_structure_ckpt,
            yolo_seg_brick_gripper_ckpt=self.cfg.yolo_seg_brick_gripper_ckpt
        )

        # Get skill configs from pipeline config
        pick_skill_config = self.cfg.get_pick_skill_config()
        assemble_skill_config = self.cfg.get_assemble_skill_config()

        self.skills = self._build_skills(pick_skill_config, assemble_skill_config)

        if self.cfg.enable_step_identification:
            self.scene_aligner.configure_plan(
                plan=self.plan,
                config=self.cfg.get_step_identifier_config(),
            )

    def _build_skills(self, pick_skill_config, assemble_skill_config) -> dict:
        """Build the pick and assembly skills with their condition checks."""
        return {
            "pick": PickSkill(
                robot_interface=self.robot_interface,
                sam2_interface=self.sam2_interface,
                skill_cfg=pick_skill_config,
            ),
            "assemble": AssemblySkill(
                robot_interface=self.robot_interface,
                sam2_interface=self.sam2_interface,
                aligner_interface=self.scene_aligner,
                skill_cfg=assemble_skill_config,
            ),
        }

    def capture_scene(self):
        """One fixed-camera frame, cropped the way the skills see it."""
        return self.robot_interface.get_observation()["fixed_camera"]

    def identify_current_step(self) -> StepIdentification:
        """Identify the assembly state from the current fixed-camera view."""
        if not self.cfg.enable_step_identification:
            raise RuntimeError("Step identification is disabled")
        return self.scene_aligner.identify(self.capture_scene())

    def _print_identification(self, ident: StepIdentification) -> None:
        name = self.__class__.__name__
        best_state = self.plan.num_steps + 1 if ident.is_complete else ident.step_id
        print(f"[{name}] Bench reads: {ident.summary()}")
        for score in ident.scores:
            label = ("completed structure" if score.step_id == self.plan.num_steps + 1
                     else f"before brick {score.step_id}")
            marker = "  <-- best" if score.step_id == best_state else ""
            rejected = f"  REJECTED ({score.reject_reason})" if score.rejected else ""
            print(f"[{name}]   state {score.step_id:>2} ({label:<20}) "
                  f"IoU {score.iou:.3f}  scale {score.scale:.3f}{rejected}{marker}")

    def _build_step(self, step_id: int) -> bool:
        """Run the pick and assembly skills for the same plan step."""
        name = self.__class__.__name__
        assembly_step = self.plan.get_step(step_id)

        success, _ = self.skills["pick"].run({"assembly_step": assembly_step})
        if not success:
            print(f"[{name}] Pick skill failed at step {step_id}.")
            return False

        success, _ = self.skills["assemble"].run({"assembly_step": assembly_step})
        if not success:
            print(f"[{name}] Assemble skill failed at step {step_id}.")
            return False

        return True

    def _confirm_continue(self) -> bool:
        """Ask how to proceed after a failed skill."""
        action = ("re-identify the current state" if self.cfg.enable_step_identification
                  else "skip this step and continue to the next")
        while True:
            try:
                answer = input(f"Skill failed. Continue and {action}? [y/n]: ").strip().lower()
            except EOFError:
                return False
            if answer in ("y", "n"):
                return answer == "y"
            print("Please enter y or n.")

    def run(self):
        name = self.__class__.__name__
        step_id = 1
        try:
            while True:
                if self.cfg.enable_step_identification:
                    ident = self.identify_current_step()
                    self._print_identification(ident)
                    if not ident.confident or (ident.step_id is None and not ident.is_complete):
                        time.sleep(1.0)
                        continue
                    if ident.is_complete:
                        print(f"[{name}] Structure matches the finished state. Done.")
                        break
                    step_id = ident.step_id
                elif step_id > self.plan.num_steps:
                    print(f"[{name}] Reached the end of the assembly plan. Done.")
                    break

                print(f"[{name}] Building step {step_id}/{self.plan.num_steps}.")
                if not self._build_step(step_id) and not self._confirm_continue():
                    print(f"[{name}] Stopped by user.")
                    break
                if not self.cfg.enable_step_identification:
                    step_id += 1

        except KeyboardInterrupt:
            print("Pipeline interrupted by user.")
        except Exception as e:
            print(f"Pipeline encountered an error: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Clean up resources used by the pipeline.
            - Stop all policy controllers in skills
            - Close any open visualization windows
            - Disconnect robot interface if needed
        """
        print("\n[LongHorizonPipeline] Cleaning up...")

        # Stop policy controllers in each skill
        for skill_name, skill in self.skills.items():
            if hasattr(skill, 'policy_controller') and skill.policy_controller is not None:
                try:
                    if skill.policy_controller.is_running():
                        print(f"[LongHorizonPipeline] Stopping {skill_name} skill controller...")
                        skill.policy_controller.stop()
                except Exception as e:
                    print(f"[LongHorizonPipeline] Error stopping {skill_name} controller: {e}")

        # Close any OpenCV windows
        cv2.destroyAllWindows()

        # Disconnect robot interface
        self.robot_interface.disconnect()

        print("[LongHorizonPipeline] Cleanup complete.")
