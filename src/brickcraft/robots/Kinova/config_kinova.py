from dataclasses import dataclass, field
from lerobot.robots.config import RobotConfig
import numpy as np

"""
Configuration for Kinova Gen3 Base Class. Shared by all variants.
"""


@dataclass
class KinovaBaseConfig(RobotConfig):
    topics: dict = field(default_factory=lambda: {
        # Control topics
        "ee_pose_action": "/kinova_ros_control/cartesian_target",  # geometry_msgs::PoseStamped
        "gripper_action": "/kinova_ros_control/gripper_target",  # sensor_msgs::JointState (0.0-1.0)

        # Feedback topics
        # Use ee_pose_fk: raw Cartesian feedback has a slight inconsistency with FK.
        # "ee_pose": "/kinova_ros_control/feedback_cartesian",  # geometry_msgs::PoseStamped
        "ee_pose_fk": "/kinova_ros_control/feedback_cartesian_fk",  # geometry_msgs::PoseStamped (FK)
        "joint_state": "/kinova_ros_control/feedback_joint_states",  # sensor_msgs::JointState (rad)
        "external_wrench": "/kinova_ros_control/feedback_external_wrench",  # geometry_msgs::WrenchStamped
        "gripper_state": "/kinova_ros_control/feedback_gripper_state",  # sensor_msgs::JointState
        "wrist_camera": "/wrist_cam/camera/color/image_raw/compressed",  # sensor_msgs::CompressedImage
        "fixed_camera": "/fixed_cam/camera/color/image_raw/compressed",  # sensor_msgs::CompressedImage
    })

    @property
    def all_image_features(self) -> dict:
        return {
            "observation.images.wrist_camera": {
                "dtype": "video",
                "shape": (*self.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.fixed_camera": {
                "dtype": "video",
                "shape": (*self.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.goal_image": {
                "dtype": "video",
                "shape": (*self.image_size, 3),
                "names": ["height", "width", "channel"],
            },
        }
    @property
    def all_state_features(self) -> dict:
        return {
            "observation.state.gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "observation.state.external_force_scaled": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["fx_scaled", "fy_scaled", "fz_scaled"],
            },
            "observation.state.ee_pose_9d": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["ee_pose_x", "ee_pose_y", "ee_pose_z", "ee_pose_r11", "ee_pose_r12", "ee_pose_r13", "ee_pose_r21", "ee_pose_r22", "ee_pose_r23"],
            },
            "observation.state.onehot_encoding": {  # deprecated, will be removed in future
                "dtype": "float32",
                "shape": (5,),
                "names": ["center", "left", "right", "front", "back"],
            },
        }
    @property
    def all_action_features(self) -> dict:
        return {
            f"action.ee_pose_9d": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["ee_pose_x", "ee_pose_y", "ee_pose_z", "ee_pose_r11", "ee_pose_r12", "ee_pose_r13", "ee_pose_r21", "ee_pose_r22", "ee_pose_r23"],
            },
            f"action.gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
        }
    

    num_arm_joints: int = 7 
    fps: int = 10
    # Gripper state/action convention shared by online control and rosbag conversion.
    # A feedback position below the threshold is open; otherwise it is closed.
    gripper_closed_threshold: float = 0.4
    gripper_open_position: float = 0.0
    gripper_close_position: float = 0.8
    # Policy input size before training-time augmentation.
    image_size: tuple[int, int] = (256, 256)  # [height, width]  
    force_scale_factor: float = 0.05  # Scale external force readings
    image_crop_params: dict = field(default_factory=lambda: {
        "wrist_camera": {
            "target_h": 256,
            "target_w": 256,
            "crop_mode": "bottom_center_right",
            "scale_factor": 1.1,
            "keep_aspect": True,
        },
        "fixed_camera": {
            "target_h": 256,
            "target_w": 256,
            "crop_mode": "bottom_center_right",
            "scale_factor": 1.1,
            "keep_aspect": True,
        },
        "goal_image": {
            "target_h": 256,
            "target_w": 256,
            "crop_mode": "bottom_center_right",
            "scale_factor": 1.1,
            "keep_aspect": True,
        },
    })
    # How the segmentation masks are rendered into the policy's fixed_camera view.
    # Must match what the checkpoint was trained with.
    image_seg_params: dict = field(default_factory=lambda: {
        "mode": "background_dim",
        "dim_factor": 0.25,  # 0.25 = normal run, 0.0 = black background ablation
        "add_contours": False,
        "expand_pixels": 0,
        "fade_distance": 10,
    })
    arm_joint_limits: np.ndarray = field(default_factory=lambda: np.deg2rad(np.array([[-720.0, 720.0], # [lower limit, upper limit]
                                                [-128.9, 128.9],
                                                [-720.0, 720.0],
                                                [-147.8, 147.8],
                                                [-720.0, 720.0],
                                                [-120.3, 120.3],
                                                [-720.0, 720.0]],
                                                dtype=np.float32)))


@dataclass
class KinovaPose9DConfig(KinovaBaseConfig):
    type: str = "kinova_pose9d"
