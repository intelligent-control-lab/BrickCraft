"""Run the assembly policy for one step, with live SAM2 segmentation.

Configure the policy checkpoint, task encoding, and SAM2 weights in
configs/single_step_config.yaml. Manually prepare the existing structure on the
assembly baseplate and place the target brick in the robot's gripper.

    python scripts/run_single_step.py
    python scripts/run_single_step.py --config my_config.yaml
    python scripts/run_single_step.py --task-encoding 0 0 1 0 1 0

Press s in the console to start an episode. In the SAM2 prompt window, select
the three objects in this order:
  1. Click the reference brick in the existing structure, then press N.
  2. Click the target brick held by the robot, then press N.
  3. Click the robot gripper, then press Enter to finish selection.

Left click adds a positive point; right click adds a negative point to exclude
background or neighboring objects. Each object needs at least one positive
point. N starts the next object; add all points for the current object before
pressing it.

When confirm_segmentation is true (default), a review window shows the initial
segmentation. Press R to select the prompts again, or another key to accept and
run the policy. Setting it to false skips review; interactive prompt selection
is still required.

Console keys: q stops the current episode; e exits. Restore the structure and
the grasped target brick before starting another episode with s.
"""

import argparse
import yaml
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel

from brickcraft.utils.task_encoding_utils import validate_task_encoding


class SingleStepConfig(BaseModel):
    task_encoding: List[float]
    episode_timeout: float = 60.0
    sam2_config: Dict[str, Any]
    async_policy_controller_config: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str) -> "SingleStepConfig":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


start_episode = False
exit_event = threading.Event()


def keyboard_listener(controller):
    global start_episode
    print("Press 's' to start a new episode, press 'q' to stop the episode, press 'e' to exit.")
    while not exit_event.is_set():
        cmd = input().strip().lower()
        if cmd == "s":
            if not controller.is_running():
                start_episode = True
        elif cmd == "q":
            if controller.is_running():
                controller.stop()
                start_episode = False
            print("Episode stopped. press 's' to start a new episode.")
        elif cmd == "e":
            print("Exiting keyboard listener.")
            controller.stop()
            exit_event.set()
            break
        time.sleep(0.1)


def main():
    global start_episode

    parser = argparse.ArgumentParser(description="Run the assembly policy for a single step")
    parser.add_argument(
        "--config",
        default="./configs/single_step_config.yaml",
        help="Single-step config path (default: %(default)s)",
    )
    parser.add_argument(
        "--task-encoding", type=float, nargs=6, default=None, metavar="X",
        help="Override the config's 6D task_encoding: x y z orientation reference_type target_type",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    cfg = SingleStepConfig.from_yaml(str(config_path))
    task_encoding = args.task_encoding if args.task_encoding is not None else cfg.task_encoding
    task_encoding = validate_task_encoding(task_encoding, source="single-step task_encoding")

    # Load hardware dependencies after configuration validation.
    from sam2.build_sam import build_sam2_camera_predictor  # type: ignore
    from brickcraft.controller import AsyncPolicyController, AsyncPolicyControllerConfig
    from brickcraft.perception.live_segmenter import LiveSegmenter
    from brickcraft.pipeline.long_horizon_pipeline_config import SAM2Config
    from brickcraft.robots.Kinova import KinovaPose9D, KinovaPose9DConfig

    # Robot
    robot_config = KinovaPose9DConfig()
    robot = KinovaPose9D(robot_config)
    robot.connect()
    print(f"Robot {robot_config.type} connected.")

    # SAM2 segmenter
    sam2_cfg = SAM2Config(**cfg.sam2_config)
    sam2_predictor = build_sam2_camera_predictor(
        config_file=sam2_cfg.sam2_config,
        ckpt_path=sam2_cfg.sam2_checkpoint,
        device=sam2_cfg.sam2_device,
    )
    sam2_segmenter = LiveSegmenter(predictor=sam2_predictor)
    print("SAM2 Segmenter initialized.")

    # Policy controller
    controller = AsyncPolicyController(
        robot_interface=robot,
        sam2_interface=sam2_segmenter,
        config=AsyncPolicyControllerConfig(**cfg.async_policy_controller_config),
    )
    if controller.has_task_encoding:
        controller.set_task(np.array(task_encoding, dtype=np.float32))

    listener_thread = threading.Thread(target=keyboard_listener, args=(controller,))
    listener_thread.daemon = True
    listener_thread.start()

    try:
        while not exit_event.is_set():
            if start_episode:
                controller.run_episode(episode_timeout=cfg.episode_timeout)
                start_episode = False
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("User interrupted. Exiting...")
        exit_event.set()
        controller.stop()


if __name__ == "__main__":
    main()
