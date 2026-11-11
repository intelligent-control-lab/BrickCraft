"""
KinovaPose9D Controller with interpolated Cartesian motion control.

This controller wraps KinovaPose9D and provides high-level control interfaces
with velocity-based interpolation for smooth motion execution.
"""

import numpy as np
import time
from typing import Optional
from scipy.spatial.transform import Rotation, Slerp
from brickcraft.robots.Kinova.kinova_pose9d import KinovaPose9D
from brickcraft.utils.kinematics_utils import R_to_rot6d, rot6d_to_R


class KinovaPose9dController:   
    def __init__(
        self,
        kinova: KinovaPose9D,
        control_frequency: float = 10.0,
        linear_velocity: float = 0.05,  # m/s
        angular_velocity: float = 30,  # deg/s
    ):
        """      
        Args:
            kinova: KinovaPose9D robot interface object
            control_frequency: Control loop frequency in Hz
            linear_velocity: Linear velocity for Cartesian motions in m/s
            angular_velocity: Angular velocity for rotational motions in deg/s
        """
        self.kinova = kinova
        self.control_frequency = control_frequency
        self.control_period = 1.0 / control_frequency
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        
        
        print(
            f"KinovaPose9dController initialized: "
            f"control_freq={control_frequency}Hz, "
            f"linear_vel={linear_velocity}m/s, "
            f"angular_vel={angular_velocity}deg/s"
        )
    
    def move_cartesian6d(
        self,
        target_pose6d: np.ndarray,
        blocking: bool = True,
        linear_velocity: Optional[float] = None,
        angular_velocity: Optional[float] = None,
        timeout: float = 10.0,
    ) -> bool:
        """
        Move end-effector to target Cartesian pose with velocity-based interpolation.
        
        Args:
            target_pose6d: Target pose as [x, y, z, rx, ry, rz] where:
                        - x, y, z: position in meters
                        - rx, ry, rz: Euler angles in degrees
            blocking: If True, block until motion completes. If False, return immediately.
            linear_velocity: Override default linear velocity (m/s)
            angular_velocity: Override default angular velocity (deg/s)
            timeout: Seconds to wait for the arm to settle on the target once the
                trajectory has been sent
            
        Returns:
            True if motion completed successfully, False otherwise
        """
        if len(target_pose6d) != 6:
            Warning(f"[{self.__class__.__name__}] Invalid target_pose shape: expected (6,), got {target_pose6d.shape}")
            return False
        
        # Get current pose
        current_pose_9d = self.kinova.get_current_pose9d()
        if current_pose_9d is None:
            Warning(f"[{self.__class__.__name__}] Cannot get current pose. Robot may not be connected.")
            return False

        print(f"[{self.__class__.__name__}] Moving to target pose: {target_pose6d}")

        # Convert current 9D pose to 6D (position + euler in degrees)
        current_pos = current_pose_9d[:3]
        current_rot6d = current_pose_9d[3:9]
        current_R = rot6d_to_R(current_rot6d)
        current_euler = Rotation.from_matrix(current_R).as_euler('xyz', degrees=True)
        current_pose6d = np.concatenate([current_pos, current_euler])

        # Use provided velocities or defaults
        lin_vel = linear_velocity if linear_velocity is not None else self.linear_velocity
        ang_vel_deg = angular_velocity if angular_velocity is not None else self.angular_velocity
        ang_vel_rad = np.deg2rad(ang_vel_deg)  # Convert deg/s to rad/s
        
        # Generate interpolated trajectory
        trajectory = self._generate_trajectory(
            current_pose6d,
            target_pose6d,
            lin_vel,
            ang_vel_rad,  # Pass in rad/s
        )       
        # Execute trajectory
        if not blocking:
            print(f"[{self.__class__.__name__}] Warning: Non-blocking mode not yet implemented. Using blocking mode.")

        success = self._execute_trajectory_blocking(trajectory)
        
        if not success:
            print(f"[{self.__class__.__name__}] Trajectory execution failed")
            return False
        
        # The settle timeout starts once the trajectory has been sent, so a long
        # (or slowly executed) trajectory still gets its reached-check.
        timeout_start = time.time()

        # Define thresholds (can be adjusted based on requirements)
        position_threshold = 0.005  # 5mm
        orientation_threshold = 3.0  # 3 degrees (now in degrees for consistency)
        
       
        while time.time() - timeout_start < timeout:
            # Get current pose
            current_pose_9d = self.kinova.get_current_pose9d()
            if current_pose_9d is None:
                Warning(f"[{self.__class__.__name__}] Cannot get current pose. Robot may not be connected.")
                return False
            
            # Convert current 9D pose to 6D (in degrees)
            current_pos = current_pose_9d[:3]
            current_rot6d = current_pose_9d[3:9]
            current_R = rot6d_to_R(current_rot6d)
            current_euler = Rotation.from_matrix(current_R).as_euler('xyz', degrees=True)
            current_pose_6d = np.concatenate([current_pos, current_euler])
            
            # Calculate position and orientation errors
            pos_error = np.linalg.norm(current_pose_6d[:3] - target_pose6d[:3])
            
            # Calculate orientation error using rotation matrix (avoid euler angle discontinuity)
            current_rot = Rotation.from_euler('xyz', current_pose_6d[3:6], degrees=True)
            target_rot = Rotation.from_euler('xyz', target_pose6d[3:6], degrees=True)
            rot_diff = current_rot.inv() * target_rot # radians
            orient_error = np.rad2deg(rot_diff.magnitude())  # Convert to degrees
            
            # Check if target reached
            if pos_error < position_threshold and orient_error < orientation_threshold:
                print(f"[{self.__class__.__name__}] Motion completed successfully")
                print(f"  Position error: {pos_error*1000:.2f}mm, Orientation error: {orient_error:.2f}deg")
                print(f"  Time taken: {time.time() - timeout_start:.2f}s")
                return True

            time.sleep(self.control_period)

        # Timeout reached - get final pose for error reporting
        final_pose_9d = self.kinova.get_current_pose9d()
        if final_pose_9d is None:
            print(f"[{self.__class__.__name__}] Warning: Timeout after {timeout}s, cannot get final pose")
            return False
        
        # Convert final 9D pose to 6D (in degrees)
        final_pos = final_pose_9d[:3]
        final_rot6d = final_pose_9d[3:9]
        final_R = rot6d_to_R(final_rot6d)
        final_euler = Rotation.from_matrix(final_R).as_euler('xyz', degrees=True)
        final_pose_6d = np.concatenate([final_pos, final_euler])
        
        # Calculate final errors
        pos_error = np.linalg.norm(final_pose_6d[:3] - target_pose6d[:3])
        
        # Calculate orientation error using rotation matrix (avoid euler angle discontinuity)
        final_rot = Rotation.from_euler('xyz', final_pose_6d[3:6], degrees=True)
        target_rot = Rotation.from_euler('xyz', target_pose6d[3:6], degrees=True)
        rot_diff = final_rot.inv() * target_rot
        orient_error = np.rad2deg(rot_diff.magnitude())  # Convert to degrees
        
        print(f"[{self.__class__.__name__}] Warning: Timeout after {timeout}s, target pose not reached")
        print(f"  Position error: {pos_error*1000:.2f}mm (threshold: {position_threshold*1000:.0f}mm)")
        print(f"  Orientation error: {orient_error:.2f}deg (threshold: {orientation_threshold:.0f}deg)")
        print(f"  Current pose: {final_pose_6d}")
        print(f"  Target pose:  {target_pose6d}")
        
        return False

    def close_gripper(self, blocking: bool = False, timeout: float = 3.0) -> bool:
        """
        Close the gripper.
        
        Args:
            blocking: If True, wait until gripper closes or timeout
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if gripper closed successfully, False otherwise
        """
        print(f"[{self.__class__.__name__}] Closing gripper...")
        self.kinova.move_gripper(1)  # 1.0 = closed position

        if blocking:
            start_time = time.time()
            previous_pos = None
            stable_count = 0
            fully_closed_threshold = 0.75
            grasp_contact_threshold = 0.10
            stable_delta_threshold = 0.01
            stable_count_required = 5
            while True:
                current_pos = self.kinova.get_current_gripper_position()
                if current_pos is not None:
                    if current_pos > fully_closed_threshold:  # Nearly fully closed
                        print(f"[{self.__class__.__name__}] Gripper closed successfully")
                        return True

                    # Treat "contacted object and stopped moving" as a successful grasp.
                    # This avoids false negatives when the fingers cannot fully close
                    # because a brick is already secured between them.
                    if previous_pos is not None and abs(current_pos - previous_pos) < stable_delta_threshold:
                        stable_count += 1
                    else:
                        stable_count = 0

                    if current_pos > grasp_contact_threshold and stable_count >= stable_count_required:
                        print(
                            f"[{self.__class__.__name__}] Gripper grasped object successfully "
                            f"(stalled at position {current_pos:.3f})"
                        )
                        return True

                    previous_pos = current_pos
                
                if time.time() - start_time > timeout:
                    print(f"[{self.__class__.__name__}] Warning: Gripper close timeout after {timeout}s")
                    return False
                
                time.sleep(self.control_period)
        
        return True

    def _generate_trajectory(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        linear_velocity: float,
        angular_velocity: float,
    ) -> list[np.ndarray]:
        """
        Generate interpolated trajectory between start and end poses.
        
        Args:
            start_pose: Starting pose [x, y, z, rx, ry, rz] (euler angles in degrees)
            end_pose: Target pose [x, y, z, rx, ry, rz] (euler angles in degrees)
            linear_velocity: Linear velocity in m/s
            angular_velocity: Angular velocity in rad/s
            
        Returns:
            List of interpolated poses (9D format for direct execution)
        """
        # Separate position and orientation
        start_pos = start_pose[:3]
        end_pos = end_pose[:3]
        start_euler = start_pose[3:6]
        end_euler = end_pose[3:6]
        
        # Convert euler angles (in degrees) to rotation objects for SLERP
        start_rot = Rotation.from_euler('xyz', start_euler, degrees=True)
        end_rot = Rotation.from_euler('xyz', end_euler, degrees=True)
        
        # Calculate position and orientation differences
        pos_diff = end_pos - start_pos
        linear_distance = np.linalg.norm(pos_diff)
        
        # Calculate angular distance using rotation difference
        rot_diff = start_rot.inv() * end_rot
        angular_distance = np.linalg.norm(rot_diff.as_rotvec())
        
        # Calculate required time based on maximum of position and orientation
        time_for_position = linear_distance / linear_velocity if linear_velocity > 0 else 0
        time_for_orientation = angular_distance / angular_velocity if angular_velocity > 0 else 0
        
        total_time = max(time_for_position, time_for_orientation)
        
        # Calculate number of waypoints based on control frequency
        num_waypoints = max(int(total_time * self.control_frequency), 1)
        
        # Generate interpolated trajectory
        trajectory_9d = []
        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints if num_waypoints > 0 else 1.0
            
            # Linear interpolation for position
            interp_pos = start_pos + alpha * pos_diff
            
            # SLERP (Spherical Linear Interpolation) for orientation
            # Using scipy's Slerp for smooth rotation interpolation
            key_rots = Rotation.from_quat([start_rot.as_quat(), end_rot.as_quat()])
            slerp = Slerp([0, 1], key_rots)
            interp_rot = slerp([alpha])[0]
            
            # Convert rotation to rot6d
            interp_R = interp_rot.as_matrix()
            interp_rot6d = R_to_rot6d(interp_R)
            
            # Combine position and rotation into 9D pose
            pose_9d = np.concatenate([interp_pos, interp_rot6d])
            trajectory_9d.append(pose_9d)
        
        return trajectory_9d
    
    def _execute_trajectory_blocking(self, trajectory: list[np.ndarray]) -> bool:
        """
        Execute trajectory in blocking mode at fixed control frequency.
        
        Args:
            trajectory: List of waypoints in 9D pose format
            
        Returns:
            True if trajectory executed successfully, False if interrupted
        """
        
        try:
            for i, waypoint in enumerate(trajectory):
                loop_start_time = time.time()
                
                # Send waypoint to robot
                self.kinova.move_pose9d(waypoint)
                               
                # Sleep to maintain control frequency
                elapsed = time.time() - loop_start_time
                sleep_time = self.control_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            return True
            
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error during trajectory execution: {e}")
            return False
