"""SpaceMouse intervention device for Kinova absolute Pose9D control."""

from __future__ import annotations

import numpy as np

from brickcraft.teleop.spacemouse_teleop_common import (
    POSE9D_NAMES,
    SpaceMouseTeleop,
    TeleopConfig,
    integrate_pose_delta,
    pose9d_to_action_dict,
    pose9d_to_transform,
    transform_to_pose9d,
)


class SpacemousePose9DInterventionDevice:
    """Integrate SpaceMouse deltas into bounded Kinova Pose9D actions."""

    def __init__(self, config: TeleopConfig) -> None:
        self.teleop = SpaceMouseTeleop(config)
        self._target_pose = np.eye(4, dtype=float)
        self._gripper_position = config.gripper_open_position
        self._synced = False

    def update_mouse_state(self, state) -> None:
        self.teleop.update_mouse_state(state)

    def reset_input_state(self) -> None:
        self.teleop.reset_input_state()

    def sync_to_observation(self, observation: dict[str, object]) -> None:
        pose9d = self._pose9d_from_observation(observation)
        self._target_pose = pose9d_to_transform(pose9d)
        self._gripper_position = float(
            np.clip(
                observation.get("gripper_position", self._gripper_position),
                0.0,
                1.0,
            )
        )
        self._synced = True

    def get_action(
        self, observation: dict[str, object]
    ) -> tuple[dict[str, float], list[str]]:
        if not self._synced:
            self.sync_to_observation(observation)

        delta, events = self.teleop.step(self.teleop.consume_mouse_state())
        self._target_pose = integrate_pose_delta(self._target_pose, delta)
        current_pose9d = self._pose9d_from_observation(observation)
        self._target_pose = self.teleop.clamp_target_to_observation(
            self._target_pose, current_pose9d
        )
        for event in events:
            if event == "gripper_toggle":
                self._toggle_gripper(current_pose9d)

        return (
            pose9d_to_action_dict(
                transform_to_pose9d(self._target_pose), self._gripper_position
            ),
            events,
        )

    def _toggle_gripper(self, current_pose9d: np.ndarray) -> None:
        config = self.teleop.config
        if self._gripper_position >= config.gripper_toggle_threshold:
            # Hold the measured pose while initiating an open action. Releasing
            # an object can move the arm, so do not resend a stale pose target.
            self._target_pose = pose9d_to_transform(current_pose9d)
            self._gripper_position = config.gripper_open_position
        else:
            self._gripper_position = config.gripper_close_position
        self._gripper_position = float(np.clip(self._gripper_position, 0.0, 1.0))

    @staticmethod
    def _pose9d_from_observation(observation: dict[str, object]) -> np.ndarray:
        try:
            return np.array([observation[name] for name in POSE9D_NAMES], dtype=np.float32)
        except KeyError as exc:
            raise KeyError(f"Observation is missing Pose9D key {exc}.") from exc
