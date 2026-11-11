from lerobot.robots.robot import Robot
from brickcraft.robots.Kinova.config_kinova import KinovaBaseConfig
from brickcraft.utils.image_utils import *
from brickcraft.utils.kinematics_utils import *

# ROS Interface
import rospy

import numpy as np
from functools import cached_property
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod



class KinovaBase(Robot, ABC):
    """
    Kinova Base Class

    The observations and actions should be specified in subclasses by defining self.image_keys, self.state_keys, and self.action_keys.
    """

    config_class = KinovaBaseConfig
    name = "kinova_base"

    def __init__(self, config: KinovaBaseConfig) -> None:
        super().__init__(config)
        self.config = config
        required_attrs = ["topics", 
                          "all_image_features",
                          "all_state_features", 
                          "all_action_features", 
                          "num_arm_joints", 
                          "image_size",
                          "fps", 
                          "force_scale_factor",
                          "arm_joint_limits",
                          "image_crop_params"]
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise AttributeError(f"robot config is missing required attribute: {attr}")
            
        """ Define Observations and Actions """
        # Initialize as empty, must be specified by subclasses
        self.image_keys = []
        self.state_keys = []
        self.action_keys = []

        """ ROS Interfaces """
        self.required_topics = []
        self.topic_timestamps = {}  # record timestamps of latest messages from each topic

        """ Internal States """
        self._is_connected = False
        self._is_calibrated = False

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = {}

        for key in self.image_keys:
            shape = self.config.all_image_features[f"observation.images.{key}"]["shape"]
            features[key] = shape  # (H, W, 3)

        for state_key in self.state_keys:
            names = self.config.all_state_features[state_key]["names"]
            for name in names:
                features[name] = float

        return features


    @cached_property
    def action_features(self) -> dict[str, type]:
        features = {}
        for action_key in self.action_keys:
            names = self.config.all_action_features[action_key]["names"]
            for name in names:
                features[name] = float
        return features

    @cached_property
    def state_dimension(self) -> int:
        dim = 0
        for key in self.state_keys:
            shape = self.config.all_state_features[key]["shape"]
            dim += shape[0]
        return dim

    @cached_property
    def action_dimension(self) -> int:
        dim = 0
        for key in self.action_keys:
            shape = self.config.all_action_features[key]["shape"]
            dim += shape[0]
        return dim

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        pass

    def connect(self, calibrate: bool = True) -> None:
        all_ok = True
        publisher_topics = rospy.get_published_topics()
        for topic in self.required_topics:
            if not any(t[0] == topic for t in publisher_topics):
                print(f"Required topic '{topic}' is not being published. Robot connection failed.")
                all_ok = False
        self._is_connected = all_ok

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def calibrate(self) -> None:
        return
    
    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def configure(self) -> None:
        return

    def disconnect(self) -> None:
        self._is_connected = False





