from .kinova_base import KinovaBase
from .config_kinova import KinovaPose9DConfig
from brickcraft.utils.image_utils import *
from brickcraft.utils.kinematics_utils import *

# ROS Interface
import rospy
from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import PoseStamped, WrenchStamped

import numpy as np
import cv2
import threading
from typing import Any, Dict, Optional
from pathlib import Path


class KinovaPose9D(KinovaBase):

    config_class = KinovaPose9DConfig
    name = "kinova_pose9d"

    def __init__(self, config: KinovaPose9DConfig) -> None:
        super().__init__(config)

        """ Observations and Actions """
        self.image_keys = [  # "observation.images." will be prefixed in observation_features
            "wrist_camera",
            "fixed_camera",
        ]
        # Since the state needs to be concatenated together, the order of state_keys is very important!
        self.state_keys = [
            "observation.state.ee_pose_9d",
            "observation.state.gripper_position",
            "observation.state.external_force_scaled",
        ]
        self.action_keys = [
            "action.ee_pose_9d",
            "action.gripper_position",
        ]

        self._lock = threading.Lock()

        """ ROS Interfaces """
        self.image_subscribers = {}
        for key in self.image_keys:
            self.required_topics.append(self.config.topics[key])
            self.image_subscribers[key] = rospy.Subscriber(
                self.config.topics[key],
                CompressedImage,
                lambda msg, name=key: self._camera_callback(msg, name),
                queue_size=3,
            )
        # ee_pose subscriber
        self.ee_pose_subscriber = rospy.Subscriber(
            self.config.topics["ee_pose_fk"],  # Using FK calculated ee_pose instead of raw feedback
            PoseStamped,
            self._pose_callback,
            queue_size=10,
        )
        self.required_topics.append(self.config.topics["ee_pose_fk"])
        self.topic_timestamps["ee_pose_fk"] = 0.0
        # gripper state subscriber
        self.gripper_state_subscriber = rospy.Subscriber(
            self.config.topics["gripper_state"],
            JointState,
            self._gripper_callback,
            queue_size=10,
        )
        self.required_topics.append(self.config.topics["gripper_state"])
        self.topic_timestamps["gripper_state"] = 0.0
        # external wrench subscriber
        self.external_wrench_subscriber = rospy.Subscriber(
            self.config.topics["external_wrench"],
            WrenchStamped,
            self._wrench_callback,
            queue_size=10,
        )
        self.required_topics.append(self.config.topics["external_wrench"])
        self.topic_timestamps["external_wrench"] = 0.0
        # ee_pose action publisher
        self.ee_pose_action_publisher = rospy.Publisher(
            self.config.topics["ee_pose_action"],
            PoseStamped,
            queue_size=10
        )
        # gripper action publisher
        self.gripper_action_publisher = rospy.Publisher(
            self.config.topics["gripper_action"],
            JointState,
            queue_size=10
        )

        """ Internal States """
        # images are [H, W, C] in RGB format
        self._images = {key: None for key in self.image_keys}  # type: Dict[str, Optional[np.ndarray]]
        self._ee_pose_9d = None # type: Optional[np.ndarray]
        self._ee_action_9d = None # type: Optional[np.ndarray]
        self._external_force = None # type: Optional[np.ndarray]
        self._gripper_position = None # type: Optional[np.ndarray]
        self._gripper_action = None # type: Optional[np.ndarray]

        # Gripper action filter to prevent frequent open/close
        self._gripper_action_filter_enabled = True
        self._gripper_filter_threshold = self.config.gripper_closed_threshold
        self._gripper_open_position = self.config.gripper_open_position
        self._gripper_close_position = self.config.gripper_close_position
        self._last_sent_gripper_action = None  # Last gripper action actually sent to robot
        
        # Consecutive action count requirement
        self._gripper_consecutive_count_required = 3  # Number of consecutive same actions needed to trigger execution
        self._gripper_action_history = []  # History of recent action types: ["open", "close", ...]
        
        # Gripper action blocker to prevent rapid open-close cycles
        self._gripper_action_block_duration = 1.0  # Duration (seconds) to block any action after executing one
        self._last_gripper_action_time = None  # Timestamp of last gripper action execution
        self._last_gripper_action_type = None  # Type of last action: "open" or "close"

        if not rospy.core.is_initialized():
            rospy.init_node("kinova_pose9d", anonymous=True)

    def get_observation(self) -> dict[str, Any]:
        obs = {}
        current_timestamp = rospy.get_rostime().to_sec()
        for key in self.image_keys:
            if self._images[key] is None:
                raise RuntimeError(f"Image for '{key}' not yet received from robot.")
            obs[key] = self._images[key] 
            # Logging observation delay warnings
            # if current_timestamp - self.topic_timestamps[key] > 1.0 / self.config.fps:
            #     print(f"Warning: Observation '{key}' is delayed by {current_timestamp - self.topic_timestamps[key]:.2f} seconds.")
        for state_key in self.state_keys:
            names = self.config.all_state_features[state_key]["names"]
            arr = None
            if state_key == "observation.state.ee_pose_9d":
                arr = self._ee_pose_9d
            elif state_key == "observation.state.gripper_position":
                arr = self._gripper_position
            elif state_key == "observation.state.external_force_scaled":
                arr = self._external_force * self.config.force_scale_factor
            else:
                raise KeyError(f"Unsupported state key for {self.name}: {state_key}")
            
            if arr is None:
                raise RuntimeError(f"State for '{state_key}' not yet received from robot.")
            
            for i, name in enumerate(names):
                obs[name] = float(arr[i])
            # Logging observation delay warnings
            # if current_timestamp - self.topic_timestamps["right_arm_joint_state"] > 1.0 / self.config.fps:
            #     print(f"Warning: Observation '{state_key}' is delayed by {current_timestamp - self.topic_timestamps['right_arm_joint_state']:.2f} seconds.")
        return obs
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected:
            raise RuntimeError("Robot is not connected. Call 'connect()' before sending actions.")
        sent_action = {}
        """ Send action to robot """
        for action_key in self.action_keys:
            if action_key == "action.ee_pose_9d":
                names = self.config.all_action_features[action_key]["names"]
                try:
                    values = np.array([action[name] for name in names], dtype=np.float32)
                except KeyError as e:
                    raise KeyError(f"Missing required action key {e} for {action_key}. Expected keys: {names}")
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.get_rostime()
                pose_msg.pose.position.x = values[0]
                pose_msg.pose.position.y = values[1]
                pose_msg.pose.position.z = values[2]
                rot6d = values[3:]
                quat = rot6d_to_quat(rot6d)
                pose_msg.pose.orientation.x = quat[0]
                pose_msg.pose.orientation.y = quat[1]
                pose_msg.pose.orientation.z = quat[2]
                pose_msg.pose.orientation.w = quat[3]
                self.ee_pose_action_publisher.publish(pose_msg)
                for i, name in enumerate(names):
                    sent_action[name] = float(values[i])
            elif action_key == "action.gripper_position":
                names = self.config.all_action_features[action_key]["names"]
                try:
                    values = np.array([action[name] for name in names], dtype=np.float32)
                except KeyError as e:
                    raise KeyError(f"Missing required action key {e} for {action_key}. Expected keys: {names}")
                gripper_position = values[0]

                # Clip gripper position to [0, 1] and warn if out of range
                if gripper_position < 0.0 or gripper_position > 1.0:
                    rospy.logwarn(f"Gripper position {gripper_position:.4f} is out of range [0, 1]. Clipping to valid range.")
                    gripper_position = np.clip(gripper_position, 0.0, 1.0)
                
                # Apply gripper action filter
                if self._gripper_action_filter_enabled:
                    filtered_gripper_action = self._filter_gripper_action(gripper_position)
                    if filtered_gripper_action is not None:
                        # Filter approved: send the action
                        gripper_msg = JointState()
                        gripper_msg.header.stamp = rospy.get_rostime()
                        gripper_msg.position = [filtered_gripper_action]
                        self.gripper_action_publisher.publish(gripper_msg)
                        sent_action[names[0]] = float(filtered_gripper_action)
                    else:
                        # Filter rejected: use last sent action for logging
                        if self._last_sent_gripper_action is not None:
                            sent_action[names[0]] = float(self._last_sent_gripper_action)
                        else:
                            sent_action[names[0]] = float(gripper_position)
                else:
                    # Filter disabled: send directly
                    gripper_msg = JointState()
                    gripper_msg.header.stamp = rospy.get_rostime()
                    gripper_msg.position = [gripper_position]
                    self.gripper_action_publisher.publish(gripper_msg)
                    sent_action[names[0]] = float(gripper_position)
                    self._last_sent_gripper_action = gripper_position
            else:
                raise KeyError(f"Unsupported action key for {self.name}: {action_key}")
        
        
        return sent_action

    def _camera_callback(self, msg: CompressedImage, cam_name: str) -> None:
        """ Callback for CompressedImage messages. """
        with self._lock:
            # Avoid cv_bridge's Python/NumPy-specific binary extension.
            img = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot decode compressed image for '{cam_name}'")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
            crop_params = self.config.image_crop_params[cam_name]
            self._images[cam_name] = resize_and_crop(
                img,
                crop_params["target_h"],
                crop_params["target_w"],
                crop_mode=crop_params["crop_mode"],
                scale_factor=crop_params["scale_factor"],
                keep_aspect=crop_params["keep_aspect"],
            )
            last_ts = self.topic_timestamps.get(cam_name, 0.0)
            if hasattr(msg, 'header'):
                self.topic_timestamps[cam_name] = msg.header.stamp.to_sec()
            else:
                self.topic_timestamps[cam_name] = rospy.get_rostime().to_sec()
            # if self.topic_timestamps[cam_name] - last_ts > 1.5 / self.config.fps: 
            #     print(f"Warning: Camera '{cam_name}' message update interval {self.topic_timestamps[cam_name] - last_ts:.2f} seconds exceeds control cycle {1.0 / self.config.fps:.2f} seconds.")

    def _pose_callback(self, msg: PoseStamped) -> None:
        """ Callback for PoseStamped messages. """
        with self._lock:
            position = msg.pose.position
            orientation = msg.pose.orientation
            rot6d = quat_to_rot6d(np.array([orientation.x, orientation.y, orientation.z, orientation.w], dtype=np.float32))
            pos_array = np.array([position.x, position.y, position.z], dtype=np.float32)
            ee_pose_9d = np.concatenate([pos_array, rot6d], axis=0)
            self._ee_pose_9d = ee_pose_9d
            last_ts = self.topic_timestamps.get("ee_pose_fk", 0.0)
            self.topic_timestamps["ee_pose_fk"] = msg.header.stamp.to_sec()
            # if self.topic_timestamps["ee_pose_fk"] - last_ts > 1.5 / self.config.fps: 
            #     print(f"Warning: EE pose message update interval {self.topic_timestamps['ee_pose_fk'] - last_ts:.2f} seconds exceeds control cycle {1.0 / self.config.fps:.2f} seconds.")
    
    def _gripper_callback(self, msg: JointState) -> None:
        """ Callback for JointState messages for gripper state. """
        with self._lock:
            if len(msg.position) > 0:
                self._gripper_position = np.array([msg.position[0]], dtype=np.float32)
            last_ts = self.topic_timestamps.get("gripper_state", 0.0)
            self.topic_timestamps["gripper_state"] = msg.header.stamp.to_sec()
            # if self.topic_timestamps["gripper_state"] - last_ts > 1.5 / self.config.fps: 
            #     print(f"Warning: Gripper state message update interval {self.topic_timestamps['gripper_state'] - last_ts:.2f} seconds exceeds control cycle {1.0 / self.config.fps:.2f} seconds.")

    def _wrench_callback(self, msg: WrenchStamped) -> None:
        """ Callback for WrenchStamped messages for external force. """
        with self._lock:
            force = msg.wrench.force
            self._external_force = np.array([force.x, force.y, force.z], dtype=np.float32)
            last_ts = self.topic_timestamps.get("external_wrench", 0.0)
            self.topic_timestamps["external_wrench"] = msg.header.stamp.to_sec()
            # if self.topic_timestamps["external_wrench"] - last_ts > 1.5 / self.config.fps: 
            #     print(f"Warning: External wrench message update interval {self.topic_timestamps['external_wrench'] - last_ts:.2f} seconds exceeds control cycle {1.0 / self.config.fps:.2f} seconds.")

    def _filter_gripper_action(self, new_action: float) -> Optional[float]:
        """
        Filter gripper actions to prevent frequent open/close jittering.
        
        Requires N consecutive same actions before execution. Once executed, enables
        time-based blocking to prevent rapid state changes.
        
        Args:
            new_action: New gripper position command [0, 1]
            
        Returns:
            Configured gripper open/close position, or None if the action should be blocked.
        """
        # Convert continuous action to discrete action type
        if new_action >= self._gripper_filter_threshold:
            desired_action_type = "close"
            desired_action = self._gripper_close_position
        else:
            desired_action_type = "open"
            desired_action = self._gripper_open_position
        
        # Check if action is blocked by time-based blocker (after previous execution)
        current_time = rospy.get_rostime().to_sec()
        if self._last_gripper_action_time is not None:
            time_since_last_action = current_time - self._last_gripper_action_time
            if time_since_last_action < self._gripper_action_block_duration:
                # Within blocking period - block ANY action
                rospy.logdebug(f"Blocking {desired_action_type} action: {time_since_last_action:.2f}s since last action (block duration: {self._gripper_action_block_duration}s)")
                # Clear history during blocking period to require fresh consecutive count
                self._gripper_action_history.clear()
                return None
        
        # Update action history
        self._gripper_action_history.append(desired_action_type)
        
        # Keep only the last N actions in history
        if len(self._gripper_action_history) > self._gripper_consecutive_count_required:
            self._gripper_action_history.pop(0)
        
        # Check if we have enough consecutive same actions
        if len(self._gripper_action_history) < self._gripper_consecutive_count_required:
            rospy.logdebug(f"Accumulating {desired_action_type} actions: {len(self._gripper_action_history)}/{self._gripper_consecutive_count_required}")
            return None
        
        # Check if all actions in history are the same type
        if not all(action == desired_action_type for action in self._gripper_action_history):
            rospy.logdebug(f"Inconsistent action history: {self._gripper_action_history}, waiting for {self._gripper_consecutive_count_required} consecutive {desired_action_type}")
            return None
        
        # Only send if different from last sent action (avoid redundant commands)
        if self._last_sent_gripper_action is not None and abs(desired_action - self._last_sent_gripper_action) < 0.01:
            rospy.logdebug(f"Action {desired_action_type} already at target position, skipping")
            return None
        
        # Action approved - execute and activate time blocker
        self._last_sent_gripper_action = desired_action
        self._last_gripper_action_time = current_time
        self._last_gripper_action_type = desired_action_type
        
        # Clear history after successful execution to require fresh count for next action
        self._gripper_action_history.clear()
        
        rospy.loginfo(f"Gripper action executed: {desired_action_type} (position: {desired_action:.2f}) after {self._gripper_consecutive_count_required} consecutive commands")
        
        return desired_action

    """
    More straight-forward control interfaces
    """
    def get_current_pose9d(self) -> Optional[np.ndarray]:
        """ Get current end-effector pose in 9D format. """
        return self._ee_pose_9d
    
    def get_current_gripper_position(self) -> Optional[float]:
        """ Get current gripper position. """
        if self._gripper_position is not None:
            return float(self._gripper_position[0])
        return None
    
    def move_pose9d(self, pose_9d: np.ndarray) -> None:
        if not self._is_connected:
            raise RuntimeError("Robot is not connected. Call 'connect()' before sending actions.")
        
        position = pose_9d[0:3]
        rot6d = pose_9d[3:9]
        quat = rot6d_to_quat(rot6d)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.get_rostime()
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])

        self.ee_pose_action_publisher.publish(pose_msg)

    def move_gripper(self, position: float) -> None:
        if not self._is_connected:
            raise RuntimeError("Robot is not connected. Call 'connect()' before sending actions.")
        
        # Clip gripper position to [0, 1] and warn if out of range
        if position < 0.0 or position > 1.0:
            rospy.logwarn(f"Gripper position {position:.4f} is out of range [0, 1]. Clipping to valid range.")
            position = np.clip(position, 0.0, 1.0)
        
        gripper_msg = JointState()
        gripper_msg.header.stamp = rospy.get_rostime()
        gripper_msg.position = [float(position)]
        self.gripper_action_publisher.publish(gripper_msg)
