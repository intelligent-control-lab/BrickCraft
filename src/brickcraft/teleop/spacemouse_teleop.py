"""Kinova Gen3 SpaceMouse teleoperation over the BrickCraft ROS1 interface.

button[0]: toggle gripper; button[1]: toggle recording with --record-rosbag;
q or Ctrl+C: quit. Use --dry-run to inspect targets without publishing actions.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import logging
from pathlib import Path
import threading
import time

import numpy as np

from brickcraft.teleop import (
    BagRecorder,
    KeyboardListener,
    SpacemousePose9DInterventionDevice,
    TeleopConfig,
    open_spacemouse,
    resolve_record_topics_kinova,
)


logger = logging.getLogger(__name__)


def _wait_for_observation(robot, timeout: float, ros) -> dict:
    deadline = time.monotonic() + timeout
    last_error = None
    while not ros.is_shutdown() and time.monotonic() < deadline:
        try:
            return robot.get_observation()
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for a complete robot observation: {last_error}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-path", help="SpaceMouse /dev/hidraw* path (default: auto-detect)")
    parser.add_argument("--publish-hz", type=int, default=10, help="Control frequency in Hz")
    parser.add_argument(
        "--scale-pos", type=float, nargs=3, default=(0.05, 0.05, 0.05),
        help="Translation gains in m/s at full deflection, per base-frame axis",
    )
    parser.add_argument(
        "--scale-rot", type=float, nargs=3, default=(0.025, 0.025, 0.025),
        help="Rotation gains in rad/s at full deflection, per base-frame axis",
    )
    parser.add_argument("--deadzone", type=float, default=0.05)
    parser.add_argument(
        "--pose-error-limits", type=float, nargs=6,
        default=(0.007, 0.007, 0.010, 0.05, 0.05, 0.05),
        metavar=("DX", "DY", "DZ", "RX", "RY", "RZ"),
        help="Target error limits: translation in m, local rotation vector in rad",
    )
    parser.add_argument("--observation-timeout", type=float, default=15.0)
    parser.add_argument("--record-rosbag", action="store_true")
    parser.add_argument("--rosbag-output-dir", type=Path, default=Path("data/rosbags/teleop"))
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Read ROS feedback and SpaceMouse input without publishing actions",
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.observation_timeout) or args.observation_timeout <= 0:
        parser.error("--observation-timeout must be finite and positive")
    try:
        SpacemousePose9DInterventionDevice(_make_teleop_config(args))
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _make_teleop_config(args: argparse.Namespace) -> TeleopConfig:
    return TeleopConfig(
        device_path=args.device_path,
        publish_hz=args.publish_hz,
        scale_pos=np.asarray(args.scale_pos, dtype=float),
        scale_rot=np.asarray(args.scale_rot, dtype=float),
        deadzone=args.deadzone,
        pose_error_limits=np.asarray(args.pose_error_limits, dtype=float),
    )


def run(args: argparse.Namespace) -> None:
    # Keep ROS and robot dependencies out of imports and --help.
    import rospy
    from brickcraft.robots.Kinova.config_kinova import KinovaPose9DConfig
    from brickcraft.robots.Kinova.kinova_pose9d import KinovaPose9D

    teleop_config = _make_teleop_config(args)
    if not rospy.core.is_initialized():
        rospy.init_node("brickcraft_spacemouse_teleop", anonymous=True)

    robot_config = KinovaPose9DConfig(fps=args.publish_hz)
    # Use the current robot config's normalized gripper convention.
    teleop_config.gripper_open_position = robot_config.gripper_open_position
    teleop_config.gripper_close_position = robot_config.gripper_close_position
    teleop_config.gripper_toggle_threshold = robot_config.gripper_closed_threshold
    teleop_device = SpacemousePose9DInterventionDevice(teleop_config)
    stop_reader = threading.Event()
    reader_errors: list[Exception] = []

    with ExitStack() as cleanup:
        robot = KinovaPose9D(robot_config)
        cleanup.callback(robot.disconnect)
        robot.connect()
        if not robot.is_connected:
            raise RuntimeError("Kinova ROS topics are incomplete; robot connection failed.")
        teleop_device.sync_to_observation(
            _wait_for_observation(robot, args.observation_timeout, rospy)
        )

        spacemouse = open_spacemouse(teleop_config)
        cleanup.callback(spacemouse.close)
        recorder = None
        if args.record_rosbag:
            recorder = BagRecorder(
                args.rosbag_output_dir, resolve_record_topics_kinova(robot_config)
            )
            cleanup.callback(recorder.stop)
        keyboard = KeyboardListener()
        cleanup.callback(keyboard.stop)

        def read_spacemouse() -> None:
            while not stop_reader.is_set() and not rospy.is_shutdown():
                try:
                    state = spacemouse.read()
                    if state is None:
                        raise RuntimeError("SpaceMouse disconnected or returned no state.")
                    teleop_device.update_mouse_state(state)
                except Exception as exc:
                    reader_errors.append(exc)
                    stop_reader.set()
                    return
                stop_reader.wait(0.001)

        reader_thread = threading.Thread(target=read_spacemouse, daemon=True)
        reader_thread.start()
        cleanup.callback(reader_thread.join, timeout=1.0)
        cleanup.callback(stop_reader.set)
        keyboard.start()
        logger.info("Teleoperation active. button[0]=gripper, button[1]=rosbag, q/Ctrl+C=quit")

        period = 1.0 / args.publish_hz
        last_dry_run_log = 0.0
        while not rospy.is_shutdown() and not stop_reader.is_set():
            loop_start = time.monotonic()
            if (keyboard.poll_event() or "").lower() == "q":
                break
            if recorder is not None:
                recorder.check_running()
            observation = robot.get_observation()
            action, events = teleop_device.get_action(observation)
            for event in events:
                if event == "record_toggle" and recorder is not None:
                    if recorder.process is None:
                        logger.info("Started rosbag: %s", recorder.start())
                    else:
                        recorder.stop()
                        logger.info("Stopped rosbag")
            if stop_reader.is_set() or rospy.is_shutdown():
                break
            if args.dry_run:
                if time.monotonic() - last_dry_run_log >= 1.0:
                    logger.info("Dry-run target: %s", action)
                    last_dry_run_log = time.monotonic()
            else:
                robot.send_action(action)
            stop_reader.wait(max(0.0, period - (time.monotonic() - loop_start)))

        if reader_errors:
            raise RuntimeError(
                "SpaceMouse read failed; teleoperation stopped."
            ) from reader_errors[0]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        run(args)
    except KeyboardInterrupt:
        logger.info("Teleoperation stopped.")


if __name__ == "__main__":
    main()
