"""Shared SpaceMouse pose integration and ROS1 recording utilities."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation


POSE9D_NAMES = [
    "ee_pose_x",
    "ee_pose_y",
    "ee_pose_z",
    "ee_pose_r11",
    "ee_pose_r12",
    "ee_pose_r13",
    "ee_pose_r21",
    "ee_pose_r22",
    "ee_pose_r23",
]


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_matrix()


def _matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(matrix).as_rotvec()


@dataclass
class TeleopConfig:
    """SpaceMouse gains, mappings, and Kinova gripper positions."""

    device_path: str | None = None
    publish_hz: float = 10.0
    scale_pos: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.05, 0.05], dtype=float)
    )
    scale_rot: np.ndarray = field(
        default_factory=lambda: np.array([0.025, 0.025, 0.025], dtype=float)
    )
    deadzone: float = 0.05
    gripper_open_position: float = 0.0
    gripper_close_position: float = 0.8
    gripper_toggle_threshold: float = 0.4
    motion_mapping: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, -1],
            ],
            dtype=float,
        )
    )
    pose_error_limits: np.ndarray | None = field(
        default_factory=lambda: np.array(
            [0.003, 0.003, 0.005, 0.03, 0.03, 0.03], dtype=float
        )
    )
    # Maximum target-action error relative to the current state, ordered as
    # [x, y, z, rotation-vector x, rotation-vector y, rotation-vector z].
    # Translation limits are in metres; rotation-vector limits are in radians.


def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert BrickCraft's two-column 6D rotation to a rotation matrix."""
    values = np.asarray(rot6d, dtype=float)
    if values.shape != (6,):
        raise ValueError(f"Expected rot6d shape (6,), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("rot6d must contain finite values.")
    first = values[:3]
    second = values[3:]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-8:
        raise ValueError("Invalid rot6d: first basis vector has zero norm.")
    basis_1 = first / first_norm
    basis_2 = second - np.dot(basis_1, second) * basis_1
    second_norm = np.linalg.norm(basis_2)
    if second_norm < 1e-8:
        raise ValueError("Invalid rot6d: basis vectors are collinear.")
    basis_2 /= second_norm
    basis_3 = np.cross(basis_1, basis_2)
    return np.stack([basis_1, basis_2, basis_3], axis=1)


def pose9d_to_transform(pose9d: np.ndarray) -> np.ndarray:
    values = np.asarray(pose9d, dtype=float)
    if values.shape != (9,):
        raise ValueError(f"Expected pose9d shape (9,), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("pose9d must contain finite values.")
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = values[:3]
    transform[:3, :3] = rot6d_to_rotation_matrix(values[3:])
    return transform


def transform_to_pose9d(transform: np.ndarray) -> np.ndarray:
    values = np.asarray(transform, dtype=float)
    if values.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {values.shape}.")
    rotation = values[:3, :3]
    return np.concatenate([values[:3, 3], rotation[:, 0], rotation[:, 1]])


class SpaceMouseTeleop:
    """Convert the latest SpaceMouse sample to a bounded Cartesian delta."""

    def __init__(self, config: TeleopConfig):
        self.config = config
        self.scale_pos = np.asarray(config.scale_pos, dtype=float)
        self.scale_rot = np.asarray(config.scale_rot, dtype=float)
        self.motion_mapping = np.asarray(config.motion_mapping, dtype=float)
        self.pose_error_limits = (
            None
            if config.pose_error_limits is None
            else np.asarray(config.pose_error_limits, dtype=float)
        )
        if self.scale_pos.shape != (3,) or self.scale_rot.shape != (3,):
            raise ValueError("scale_pos and scale_rot must each contain 3 values.")
        if self.motion_mapping.shape != (6, 6):
            raise ValueError("motion_mapping must have shape (6, 6).")
        if self.pose_error_limits is not None and self.pose_error_limits.shape != (6,):
            raise ValueError("pose_error_limits must contain 6 values.")
        for name, values in (
            ("scale_pos", self.scale_pos),
            ("scale_rot", self.scale_rot),
            ("motion_mapping", self.motion_mapping),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain finite values.")
        if self.pose_error_limits is not None and (
            not np.all(np.isfinite(self.pose_error_limits)) or np.any(self.pose_error_limits < 0)
        ):
            raise ValueError("pose_error_limits must be finite and nonnegative.")
        if not np.isfinite(config.publish_hz) or config.publish_hz <= 0:
            raise ValueError("publish_hz must be finite and positive.")
        if not np.isfinite(config.deadzone) or not 0 <= config.deadzone < 1:
            raise ValueError("deadzone must be finite and in [0, 1).")
        for name in ("gripper_open_position", "gripper_close_position", "gripper_toggle_threshold"):
            value = getattr(config, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")

        self.dt = 1.0 / float(config.publish_hz)
        self._lock = threading.Lock()
        self._mouse_state = None
        self._last_consumed_state = None
        self._prev_buttons: list[bool] | None = None
        self._suppress_next_button_edges = False
        self._pending_events: list[str] = []
        self._consumed_events: list[str] = []

    def update_mouse_state(self, state: Any) -> None:
        with self._lock:
            self._mouse_state = state
            # Latch edges at HID frequency so short presses between control
            # cycles are retained. The HID driver reuses its button list.
            self._pending_events.extend(self._button_events(state))

    def consume_mouse_state(self):
        with self._lock:
            state = self._mouse_state
            self._mouse_state = None
            self._last_consumed_state = state
            self._consumed_events = self._pending_events
            self._pending_events = []
        return state

    def reset_input_state(self) -> None:
        with self._lock:
            self._mouse_state = None
            self._last_consumed_state = None
            self._prev_buttons = None
            self._suppress_next_button_edges = True
            self._pending_events.clear()
            self._consumed_events.clear()

    def _button_events(self, state: Any) -> list[str]:
        """Detect edges while holding ``_lock``; copy the driver's mutable buttons."""
        buttons = list(getattr(state, "buttons", []) or [])
        if self._suppress_next_button_edges:
            self._prev_buttons = buttons
            self._suppress_next_button_edges = False
            return []
        previous = self._prev_buttons or []
        events = [
            event
            for index, event in enumerate(("gripper_toggle", "record_toggle"))
            if index < len(buttons) and buttons[index]
            and not (index < len(previous) and previous[index])
        ]
        self._prev_buttons = buttons
        return events

    def step(self, state: Any) -> tuple[np.ndarray, list[str]]:
        if state is None:
            return np.zeros(6, dtype=float), []

        raw = np.array(
            [state.x, state.y, state.z, state.roll, state.pitch, state.yaw],
            dtype=float,
        )
        if not np.all(np.isfinite(raw)):
            raise ValueError("SpaceMouse axes must contain finite values.")
        raw[np.abs(raw) < float(self.config.deadzone)] = 0.0
        gain = np.concatenate([self.scale_pos, self.scale_rot]) * self.dt
        delta = (self.motion_mapping @ raw) * gain

        with self._lock:
            # Buffered samples were already checked by update_mouse_state().
            # Direct step() calls still support edge detection for offline use.
            if state is self._last_consumed_state:
                events = self._consumed_events
                self._consumed_events = []
                self._last_consumed_state = None
            else:
                events = self._button_events(state)
        return delta, events

    def clamp_target_to_observation(
        self, target_pose: np.ndarray, current_pose9d: np.ndarray
    ) -> np.ndarray:
        if self.pose_error_limits is None:
            return target_pose
        current = pose9d_to_transform(current_pose9d)
        result = np.asarray(target_pose, dtype=float).copy()
        position_error = result[:3, 3] - current[:3, 3]
        result[:3, 3] = current[:3, 3] + np.clip(
            position_error, -self.pose_error_limits[:3], self.pose_error_limits[:3]
        )

        rotation_error = current[:3, :3].T @ result[:3, :3]
        rotvec = _matrix_to_rotvec(rotation_error)
        clamped_rotvec = np.clip(
            rotvec, -self.pose_error_limits[3:], self.pose_error_limits[3:]
        )
        result[:3, :3] = current[:3, :3] @ _rotvec_to_matrix(clamped_rotvec)
        return result


def integrate_pose_delta(target_pose: np.ndarray, delta6: np.ndarray) -> np.ndarray:
    """Integrate translation in the base frame and rotation in the base frame."""
    delta = np.asarray(delta6, dtype=float)
    if delta.shape != (6,):
        raise ValueError(f"Expected delta6 shape (6,), got {delta.shape}.")
    result = np.asarray(target_pose, dtype=float).copy()
    result[:3, 3] += delta[:3]
    rotation_x = _rotvec_to_matrix(np.array([delta[3], 0.0, 0.0]))
    rotation_y = _rotvec_to_matrix(np.array([0.0, delta[4], 0.0]))
    rotation_z = _rotvec_to_matrix(np.array([0.0, 0.0, delta[5]]))
    result[:3, :3] = rotation_z @ rotation_y @ rotation_x @ result[:3, :3]
    return result


def pose9d_to_action_dict(pose9d: np.ndarray, gripper_pos: float) -> dict[str, float]:
    values = np.asarray(pose9d, dtype=float)
    if values.shape != (9,):
        raise ValueError(f"Expected pose9d shape (9,), got {values.shape}.")
    action = {name: float(value) for name, value in zip(POSE9D_NAMES, values, strict=True)}
    action["gripper_position"] = float(np.clip(gripper_pos, 0.0, 1.0))
    return action


class BagRecorder:
    """Start and stop a ROS1 ``rosbag record`` subprocess."""

    def __init__(self, output_dir: Path, topics: list[str]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.topics = list(dict.fromkeys(topics))
        self.process: subprocess.Popen | None = None
        self.bag_path: Path | None = None

    def start(self) -> Path:
        if self.process is not None:
            raise RuntimeError("rosbag recording is already running.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.bag_path = self.output_dir / f"rosbag_{timestamp}.bag"
        command = ["rosbag", "record", "-O", str(self.bag_path), *self.topics]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Do not stall the robot control loop while rosbag starts. The caller
        # continues checking the subprocess once per control cycle.
        self.check_running()
        return self.bag_path

    def check_running(self) -> None:
        if self.process is not None and (code := self.process.poll()) is not None:
            self.process = None
            raise RuntimeError(f"rosbag record exited with code {code}.")

    def stop(self) -> None:
        if self.process is None:
            return
        process = self.process
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=2.0)
        except ProcessLookupError:
            pass
        finally:
            self.process = None


class KeyboardListener:
    """Capture single key presses without requiring Enter."""

    def __init__(self):
        self._events: deque[str] = deque()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _listen(self) -> None:
        import select
        import termios
        import tty

        try:
            previous = termios.tcgetattr(sys.stdin)
        except termios.error:
            return
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self._stop_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if readable:
                    with self._lock:
                        self._events.append(sys.stdin.read(1))
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, previous)

    def start(self) -> bool:
        if not sys.stdin.isatty():
            return False
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def poll_event(self) -> Optional[str]:
        with self._lock:
            return self._events.popleft() if self._events else None


def open_spacemouse(config: TeleopConfig):
    """Open the SpaceMouse selected in a teleoperation config."""
    from brickcraft.teleop import pyspacemouse

    device = pyspacemouse.open(path=config.device_path)
    if device is None:
        path = config.device_path or "auto-detected path"
        raise RuntimeError(f"Failed to open SpaceMouse at {path}.")
    return device


def resolve_record_topics_kinova(robot_config=None) -> list[str]:
    """Return Kinova action, feedback, and camera topics for ROS1 recording."""
    if robot_config is None:
        from brickcraft.robots.Kinova.config_kinova import KinovaPose9DConfig

        robot_config = KinovaPose9DConfig()
    return list(dict.fromkeys(robot_config.topics.values()))
