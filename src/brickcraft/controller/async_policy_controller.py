from pathlib import Path
import brickcraft.policies  # Register before importing the patched factory.
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.robots import Robot
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame

from typing import Dict, Any, Optional, Tuple, Callable
from collections import deque
import threading
import torch
import numpy as np
import time
import cv2
from brickcraft.perception.live_segmenter import LiveSegmenter
from brickcraft.perception.mask_visualizer import MaskVisualizer
from brickcraft.controller.async_policy_controller_config import AsyncPolicyControllerConfig
from brickcraft.utils.task_encoding_utils import validate_task_encoding
import traceback


def _safe_destroy_window(window_name: str) -> None:
    """Destroy an OpenCV window, tolerating it already being gone.

    With the Qt backend, closing a window with its X button also tears down the
    GUI receiver, so waitKey() returns -1 and a later destroyWindow() raises
    "NULL guiReceiver". Swallow that instead of killing the episode.
    """
    try:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)  # let the GUI backend process the destroy event
    except cv2.error:
        pass


def _safe_destroy_all_windows() -> None:
    """Destroy all OpenCV windows, tolerating a torn-down GUI backend."""
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except cv2.error:
        pass


class TimestampedObservation:
    """Container for a raw robot observation with its timestamp."""
    def __init__(self, observation: Dict[str, Any], timestamp: float):
        self.observation = observation  # Raw robot observation (not policy format)
        self.timestamp = timestamp
        
    def __repr__(self):
        return f"TimestampedObservation(ts={self.timestamp:.3f})"


class TimestampedAction:
    """Container for an action with its expected execution timestamp."""
    def __init__(self, action: Dict[str, float], expected_timestamp: float, base_obs_timestamp: float, action_index: int):
        self.action = action
        self.expected_timestamp = expected_timestamp  # When this action should ideally be executed
        self.base_obs_timestamp = base_obs_timestamp  # Timestamp of the observation used for prediction
        self.action_index = action_index  # Index within the predicted chunk
        
    def __repr__(self):
        return f"TimestampedAction(expected_ts={self.expected_timestamp:.3f}, idx={self.action_index})"


class AsyncPolicyController:
    """
    Asynchronous Diffusion Policy Controller with threshold-based inference triggering.
    
    Key Features:
    - Dual-threaded architecture: Control thread and policy inference thread
    - Action queue with threshold-based triggering to prefetch action chunks
    - Timestamp-based action alignment to handle inference latency
    - Thread-safe observation and action queue management
    
    """
    
    def __init__(
        self,
        robot_interface: Robot,
        sam2_interface: Optional[LiveSegmenter],
        config: AsyncPolicyControllerConfig,
    ):
        """
        Initialize the Async Policy Controller.
        
        Args:
            robot_interface: Robot interface object (already connected)
            sam2_interface: SAM2 segmenter interface (optional, required if config.enable_segmentation=True)
            config: AsyncPolicyControllerConfig object
        """
        # Store robot interface
        self.robot = robot_interface
        if not self.robot.is_connected:
            raise ValueError("Robot must be connected before initializing controller!")
        print(f"Robot {self.robot.robot_type} initialized.")

        # Store config
        self.config = config
        print("\n" + "="*40)
        print(config)
        print("="*40 + "\n")
        # Load policy
        if config.policy_type != "brick_diffusion":
            raise ValueError(f"AsyncPolicyController only supports 'brick_diffusion' policy, got '{config.policy_type}'")
        self.device = config.device
        self.policy_type = config.policy_type
        policy_class = get_policy_class(config.policy_type)
        self.policy = policy_class.from_pretrained(config.pretrained_path)
        self.policy.to(self.device)
        self.policy.eval()
        state_feature = self.policy.config.input_features.get("observation.state")
        expected_shape = (self.robot.state_dimension,)
        if state_feature is None or tuple(state_feature.shape) != expected_shape:
            raise ValueError(
                f"Policy observation.state must have shape {expected_shape}, including force feedback; "
                f"got {None if state_feature is None else state_feature.shape}."
            )
        print(f"Policy loaded on device '{self.device}'.")
        print(f"Policy type: '{self.policy_type}'")
        print(f"Pretrained Path: '{config.pretrained_path}'")
        print("=" * 40)

        self._print_robot_features()
        self._print_policy_features()

        # Load preprocessors and postprocessors
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=config.pretrained_path,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": device_override},
        )

        # Task encoding support
        self.task_encoding = None
        self.has_task_encoding = "observation.task_encoding" in self.policy.config.input_features
        if self.has_task_encoding:
            print("Policy requires task_encoding. Please call set_task() before running.")

        # Controller parameters
        if config.control_freq is not None:
            self.control_freq = config.control_freq
        elif hasattr(self.robot.config, "fps"):
            self.control_freq = self.robot.config.fps
        else:
            raise ValueError("Control frequency not specified!")
        
        self.dt = 1.0 / self.control_freq
        print(f"Control frequency set to {self.control_freq} Hz (dt={self.dt:.3f}s)")

        # Get chunk size and n_obs_steps from policy config
        self.chunk_size = self.policy.config.n_action_steps
        self.n_obs_steps = self.policy.config.n_obs_steps
        if config.action_queue_thres >= self.chunk_size:
            raise ValueError(f"Action queue threshold {config.action_queue_thres} must be < chunk size {self.chunk_size}")
        self.action_queue_thres = config.action_queue_thres
        print(f"Policy Action chunk size: {self.chunk_size}")
        print(f"Inference trigger threshold: {self.action_queue_thres}")
        
        # Segmentation settings
        self.enable_segmentation = config.enable_segmentation
        self.segmentation_camera = config.segmentation_camera
        self.segmentation_output_key = (
            config.segmentation_output_key or self.segmentation_camera
        )
        self.show_segmentation = config.show_segmentation
        self.save_segmentation_masks = config.save_segmentation_masks
        self.segmentation_save_dir = Path(config.segmentation_save_dir).expanduser()
        
        # Read segmentation parameters from robot config (required if segmentation is enabled)
        if self.enable_segmentation:
            if sam2_interface is None:
                raise ValueError("sam2_interface must be provided when config.enable_segmentation=True")
            
            # Verify robot has image_seg_params
            if not hasattr(self.robot.config, "image_seg_params"):
                raise ValueError("Robot must have 'image_seg_params' attribute when segmentation is enabled")
            
            self.seg_params = self.robot.config.image_seg_params.copy()
            # Check for required 'mode' parameter
            if 'mode' not in self.seg_params:
                raise ValueError(
                    "robot.config.image_seg_params must contain a 'mode' key specifying the visualization mode "
                    "('background_dim', 'bounding_box', or 'multicolor_mask')"
                )
            
            configured_mode = self.seg_params.pop('mode')
            if config.segmentation_visualization_mode is None:
                self.visualization_mode = configured_mode
            else:
                self.visualization_mode = config.segmentation_visualization_mode
                self.seg_params = {}
            if self.segmentation_output_key not in self.robot.observation_features:
                raise ValueError(
                    f"Segmentation output key '{self.segmentation_output_key}' is not "
                    f"a robot observation image key: {list(self.robot.observation_features)}"
                )
            print(f"Segmentation mode: {self.visualization_mode}")
            print(f"Segmentation output key: {self.segmentation_output_key}")
            print(f"Segmentation parameters: {self.seg_params}")
            
            # Store segmenter interface
            self.segmenter = sam2_interface
            print(f"SAM2 segmenter interface initialized.")
        else:
            # If segmentation is not used, set empty defaults
            self.seg_params = {}
            self.visualization_mode = None
            self.segmenter = None
        
        # Segmentation state
        self.segmentation_initialized = False
        self.segmentation_save_session_dir: Optional[Path] = None
        self.segmentation_save_frame_idx = 0
        self.segmentation_save_step_id: Optional[int] = None
        if self.save_segmentation_masks:
            self.segmentation_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Segmentation masks will be saved under: {self.segmentation_save_dir}")
        
        # Latest segmentation visualization (for display in main thread)
        self.latest_segmentation_display: Optional[np.ndarray] = None
        self.segmentation_display_lock = threading.Lock()
        
        # Thread-safe data structures
        # Shadow observation buffer: stores recent n_obs_steps observations
        self.shadow_obs_buffer = deque(maxlen=self.n_obs_steps)
        self.shadow_obs_lock = threading.Lock()
        
        # Action queue: stores individual timestamped actions
        self.action_queue = deque()  # Queue of TimestampedAction objects
        self.action_queue_lock = threading.Lock()
        
        # Thread control
        self.running = False
        self.inference_thread = None
        self.inference_event = threading.Event()  # For efficient waiting
        
        # Statistics
        self.stats = {
            "total_inferences": 0,
            "total_inference_time": 0.0,  # Total time spent in inference
            "total_segmentation_time": 0.0,  # Portion of the above spent in SAM2 tracking
            "last_stats_print_time": 0.0,  # Last time stats were printed
        }
        
        # Warmup state
        self.is_warmup_complete = False
        
        self.diffusion_loss_warning_thres = config.diffusion_loss_warning_thres
        input("Please verify that robot and policy features match. Press Enter to continue...")
        print("AsyncPolicyController initialized.")

    def set_segmentation_save_context(self, step_id: Optional[int] = None) -> None:
        self.segmentation_save_step_id = step_id

    def _start_segmentation_save_session(self) -> None:
        if not self.save_segmentation_masks:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1) * 1000)
        if self.segmentation_save_step_id is not None:
            session_name = f"step_{int(self.segmentation_save_step_id):03d}_{timestamp}_{millis:03d}"
        else:
            session_name = f"episode_{timestamp}_{millis:03d}"

        self.segmentation_save_session_dir = self.segmentation_save_dir / session_name
        self.segmentation_save_session_dir.mkdir(parents=True, exist_ok=True)
        self.segmentation_save_frame_idx = 0
        print(f"Saving segmentation masks to: {self.segmentation_save_session_dir}")

    def _save_segmentation_artifacts(
        self,
        frame_rgb: np.ndarray,
        masks: Dict[Any, np.ndarray],
        masked_image_rgb: np.ndarray,
        stage: str,
    ) -> None:
        if not self.save_segmentation_masks:
            return

        if self.segmentation_save_session_dir is None:
            self._start_segmentation_save_session()
        if self.segmentation_save_session_dir is None:
            return

        if stage == "init":
            prefix = "init"
        else:
            prefix = f"track_{self.segmentation_save_frame_idx:04d}"
            self.segmentation_save_frame_idx += 1

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        masked_bgr = cv2.cvtColor(masked_image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.segmentation_save_session_dir / f"{prefix}_frame.png"), frame_bgr)
        cv2.imwrite(str(self.segmentation_save_session_dir / f"{prefix}_overlay.png"), masked_bgr)

        combined_mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
        for obj_id, mask in sorted(masks.items(), key=lambda item: str(item[0])):
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            combined_mask = np.maximum(combined_mask, mask_uint8)
            obj_name = str(obj_id).replace("/", "_")
            cv2.imwrite(
                str(self.segmentation_save_session_dir / f"{prefix}_obj_{obj_name}.png"),
                mask_uint8,
            )

        cv2.imwrite(str(self.segmentation_save_session_dir / f"{prefix}_combined.png"), combined_mask)

    def _print_policy_features(self):
        """Print policy input/output features for verification."""
        print("=" * 40)
        print("Policy Input Features:")
        for key, value in self.policy.config.input_features.items():
            print(f"  {key}: {value}")
        print("Policy Output Features:")
        for key, value in self.policy.config.output_features.items():
            print(f"  {key}: {value}")
        print("=" * 40)

    def _print_robot_features(self):
        """Print robot observation/action features for verification."""
        print("=" * 40)
        print("Robot Observation Features:")
        for key, value in self.robot.observation_features.items():
            print(f"  {key}: {value}")
        print("Robot Action Features:")
        for key, value in self.robot.action_features.items():
            print(f"  {key}: {value}")
        print("=" * 40)

    def start(self, sam2_prompts: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None):
        """
        Start the async controller (inference thread and warmup).
        
        Args:
            sam2_prompts: Optional dictionary mapping object_id to (points [shape (N, 2)], labels [shape (N,)]) tuples for SAM2 initialization.
                          If provided, the first segmentation initialization will use these prompts.
                          If None, default interactive selection will be used.
            Segmentation result review is controlled by config.confirm_segmentation.
        """
        if self.running:
            print("Controller is already running!")
            return
        
        print("\n" + "=" * 40)
        print("Starting AsyncPolicyController...")
        if sam2_prompts is not None:
            print(f"Using provided SAM2 prompts for {len(sam2_prompts)} object(s)")
        print("=" * 40)
        
        # Reset policy state
        self.policy.reset()
        
        # Clear buffers and queues
        with self.shadow_obs_lock:
            self.shadow_obs_buffer.clear()
        with self.action_queue_lock:
            self.action_queue.clear()
        
        self.is_warmup_complete = False
        
        # Reset statistics
        self.stats["total_inferences"] = 0
        self.stats["total_inference_time"] = 0.0
        self.stats["total_segmentation_time"] = 0.0
        self.stats["last_stats_print_time"] = time.time()
        
        # Start inference thread
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        self._start_segmentation_save_session()
        
        # Initialize segmentation on first frame
        if self.enable_segmentation and self.segmenter is not None:
            print("\n" + "="*60)
            if sam2_prompts is not None:
                print("Initializing object tracking with provided SAM2 prompts...")
                print(f"Prompts for {len(sam2_prompts)} object(s): {list(sam2_prompts.keys())}")
            else:
                print("Initializing object tracking...")
                print(f"Please select object(s) to track on camera: {self.segmentation_camera}")
            print("="*60)
            
            segmentation_confirmed = False
            first_initialization = True
            
            while not segmentation_confirmed:
                obs = self.robot.get_observation()
                camera_image = obs.get(self.segmentation_camera)
                
                if camera_image is None:
                    print(f"Warning: Camera '{self.segmentation_camera}' not found in observation")
                    self.segmentation_initialized = False
                    break
                else:
                    with torch.inference_mode():
                        # Use sam2_prompts for first initialization if provided, otherwise use interactive mode
                        if first_initialization and sam2_prompts is not None:
                            init_outputs = self.segmenter.initialize_tracking_with_prompts(
                                camera_image, 
                                sam2_prompts
                            )
                            first_initialization = False
                        else:
                            # Interactive mode (either no sam2_prompts or reinitializing)
                            init_outputs = self.segmenter.initialize_tracking(camera_image)
                            first_initialization = False
                        
                        if init_outputs is None:
                            print("Warning: Segmentation initialization failed")
                            self.segmentation_initialized = False
                            break
                        else:
                            mask_logits, masks = init_outputs
                            masked_image = MaskVisualizer.visualize(
                                camera_image, 
                                masks, 
                                mode=self.visualization_mode, 
                                **self.seg_params
                            )

                            print(f"✓ Segmentation initialized with {len(masks)} object(s)")
                            print(f"  Tracking objects: {list(masks.keys())}")
                            self._save_segmentation_artifacts(
                                frame_rgb=camera_image,
                                masks=masks,
                                masked_image_rgb=masked_image,
                                stage="init",
                            )
                            
                            if self.config.confirm_segmentation:
                                cv2.imshow("Segmentation Initialization", masked_image[:, :, ::-1])
                                print("  Press 'r' to reinitialize, or any other key to confirm...")
                                key = cv2.waitKey(0)
                                _safe_destroy_window("Segmentation Initialization")
                                if key in (ord('r'), ord('R')):
                                    print("  Reinitializing segmentation (interactive mode)...")
                                    continue
                                print("  Segmentation confirmed!")
                            else:
                                print("  Segmentation initialized without confirmation.")
                            self.segmentation_initialized = True
                            segmentation_confirmed = True
            
            print("="*60 + "\n")
        
        print("AsyncPolicyController started.")
        print("=" * 40 + "\n")

    def stop(self):
        """Stop the async controller gracefully."""
        if not self.running:
            return
        
        print("\nStopping AsyncPolicyController...")
        self.running = False
        self.inference_event.set()  # Wake up inference thread
        
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=2.0)
        
        # Reset segmentation state
        self.segmentation_initialized = False
        self.segmentation_save_session_dir = None
        self.segmentation_save_frame_idx = 0
        
        # NOTE: no cv2 GUI calls here. stop() is routinely invoked from a keyboard
        # listener thread, and tearing down Qt windows off the thread that created
        # them triggers "Timers cannot be started from another thread" (and can race
        # imshow in the control loop). Window teardown happens in run_episode().
        
        # Print statistics
        self._print_stats()
        print("AsyncPolicyController stopped.")

    def is_running(self) -> bool:
        """Check if controller is running."""
        return self.running

    def set_task(self, task_encoding: list[float] | np.ndarray) -> None:
        """
        Set the task encoding for the policy.
        
        Args:
            task_encoding: 6D task encoding vector (e.g., [0, 0, 1, 0, 1, 0])
        """
        if not self.has_task_encoding:
            raise ValueError(
                "A task_encoding was supplied, but this policy does not define "
                "observation.task_encoding"
            )
        
        if self.running:
            print("⚠ WARNING: Can not set task encoding while controller is running. Ignore this call.")
            return
        
        expected_shape = self.policy.config.input_features["observation.task_encoding"].shape
        task_encoding = validate_task_encoding(
            task_encoding,
            expected_shape=expected_shape,
            source="manually supplied task_encoding",
        )
        
        self.task_encoding = task_encoding
        print(f"✓ Task encoding set: {task_encoding}")

    def update_observation(self, observation: Dict[str, Any]) -> None:
        """
        Update the shadow observation buffer (called by external control loop at 10Hz).
        
        Args:
            observation: Raw observation from the robot
        """
        timestamp = time.time()
        
        # Store raw robot observation directly (no conversion to policy format yet)
        with self.shadow_obs_lock:
            timestamped_obs = TimestampedObservation(observation, timestamp)
            self.shadow_obs_buffer.append(timestamped_obs)
            
            # Check if warmup is complete
            if not self.is_warmup_complete:
                if len(self.shadow_obs_buffer) >= self.n_obs_steps:
                    self.is_warmup_complete = True
        
        # Wake up inference thread if queue is below threshold and warmup is complete
        if self.is_warmup_complete:
            with self.action_queue_lock:
                queue_size = len(self.action_queue)
            
            if queue_size < self.action_queue_thres:
                self.inference_event.set()

    def get_action(self) -> Optional[Dict[str, float]]:
        """
        Get the current action to execute (called by 10Hz control loop).
        Selects action with timestamp closest to current time and discards outdated actions.
        
        Returns:
            action: Dictionary of action values, or None if not ready
        """
        # During warmup, hold position
        if not self.is_warmup_complete:
            return None
        
        current_time = time.time()
        
        with self.action_queue_lock:
            if len(self.action_queue) == 0:
                print("⚠ WARNING: Action queue empty! Discarding action request.")
                print("  Consider increasing action_queue_thres if this happens frequently.")
                return None
                      
            # Find action with timestamp closest to current time
            best_idx = 0
            best_diff = abs(current_time - self.action_queue[0].expected_timestamp)
            
            for i in range(1, len(self.action_queue)):
                time_diff = abs(current_time - self.action_queue[i].expected_timestamp)
                if time_diff < best_diff:
                    best_diff = time_diff
                    best_idx = i
            
            if best_diff > self.dt:
                print(f"⚠ WARNING: Large time alignment error in action queue ({best_diff*1000:.1f}ms)!")
            # Remove and return the best action
            # Also remove all actions before it (they are now in the past)
            for _ in range(best_idx):
                self.action_queue.popleft()
            
            timestamped_action = self.action_queue.popleft()
            action = timestamped_action.action
        
        return action

    def _inference_loop(self):
        """
        Background inference thread loop.
        
        Monitors action queue size and triggers inference when below threshold.
        Uses shadow_obs_buffer to construct batched observations for inference.
        """
        print(f"Inference thread started (ID: {threading.get_ident()})")
        
        while self.running:
            # Check if we need to generate new actions
            with self.action_queue_lock:
                queue_size = len(self.action_queue)
            
            # Only proceed if warmup is complete and queue is below threshold
            if self.is_warmup_complete and queue_size < self.action_queue_thres:
                # Extract shadow observations
                with self.shadow_obs_lock:
                    if len(self.shadow_obs_buffer) < self.n_obs_steps:
                        # Not enough observations yet
                        continue
                    
                    # Copy observations from buffer
                    obs_list = list(self.shadow_obs_buffer)  # List of TimestampedObservation
                    base_obs_timestamp = obs_list[-1].timestamp  # Timestamp of the last observation
                
                try:
                    # Construct batch from shadow observations
                    inference_start = time.time()
                    
                    # Apply segmentation to observations if enabled
                    if self.enable_segmentation and self.segmentation_initialized:
                        obs_list = self._apply_segmentation_to_observations(obs_list)
                    
                    segmentation_duration = time.time() - inference_start
                    batch = self._construct_observation_batch(obs_list)
                    
                    # Call predict_action_chunk_new
                    action_chunk_tensor = self._predict_action_chunk_new(batch)
                                       
                    # Convert tensor to list of robot actions
                    action_chunk = self._policy_action_chunk_to_robot_actions(action_chunk_tensor)
                    
                    # Create new timestamped actions
                    new_actions = []
                    for i, action in enumerate(action_chunk):
                        # Calculate expected execution time
                        # Action i should be executed i*dt seconds after the base observation
                        expected_timestamp = base_obs_timestamp + i * self.dt
                        
                        timestamped_action = TimestampedAction(
                            action=action,
                            expected_timestamp=expected_timestamp,
                            base_obs_timestamp=base_obs_timestamp,
                            action_index=i
                        )
                        new_actions.append(timestamped_action)
                    
                    # Aggregate new actions with existing queue
                    self._aggregate_action_queue(new_actions)
                    
                    inference_duration = time.time() - inference_start

                    # Update statistics
                    self.stats["total_inferences"] += 1
                    self.stats["total_inference_time"] += inference_duration
                    self.stats["total_segmentation_time"] += segmentation_duration
                    
                    # Print statistics every 2 seconds
                    current_time = time.time()
                    time_since_last_print = current_time - self.stats["last_stats_print_time"]
                    if time_since_last_print >= 2.0:
                        avg_inference_time = self.stats["total_inference_time"] / self.stats["total_inferences"]
                        avg_segmentation_time = self.stats["total_segmentation_time"] / self.stats["total_inferences"]
                        inferences_per_sec = self.stats["total_inferences"] / time_since_last_print
                        print(
                            f"[Stats] Avg inference latency: {avg_inference_time*1000:.1f}ms "
                            f"(seg: {avg_segmentation_time*1000:.1f}ms) | "
                            f"Num of Inferences/sec: {inferences_per_sec:.2f}"
                        )
                        
                        # Reset for next period
                        self.stats["total_inferences"] = 0
                        self.stats["total_inference_time"] = 0.0
                        self.stats["total_segmentation_time"] = 0.0
                        self.stats["last_stats_print_time"] = current_time
                    
                except Exception as e:
                    print(f"✗ Inference error: {e}")
                    traceback.print_exc()
            
            # Wait for signal or timeout
            self.inference_event.wait(timeout=0.01)  # 10ms polling
            self.inference_event.clear()

    def _construct_observation_batch(self, obs_list: list[TimestampedObservation]) -> Dict[str, torch.Tensor]:
        """
        Construct a batched observation dictionary from raw robot observations.
        
        1. Converts each raw robot observation to policy format
        2. Applies preprocessing to each observation
        3. Stacks them along the temporal dimension
        
        Args:
            obs_list: List of TimestampedObservation objects with raw robot observations (length = n_obs_steps)
            
        Returns:
            Batch dictionary with temporal dimension:
            - "observation.state": (1, n_obs_steps, state_dim)
            - "observation.images": (1, n_obs_steps, num_cameras, C, H, W)
            - "observation.task_encoding": (1, n_obs_steps, task_encoding_dim) [if present]
        """
        if len(obs_list) != self.n_obs_steps:
            raise ValueError(f"Expected {self.n_obs_steps} observations, got {len(obs_list)}")
        
        # Step 1: Convert raw robot observations to policy observations
        policy_observations = []
        for ts_obs in obs_list:
            policy_obs = self._robot_observation_to_policy_observation(ts_obs.observation)
            policy_observations.append(policy_obs)
        
        # Step 2: Apply preprocessing to each policy observation
        processed_observations = [self.preprocessor(obs) for obs in policy_observations]
        
        # Step 3: Stack into batch
        batch = {}
        
        # Get keys from first observation
        sample_obs = processed_observations[0]
        
        # Separate image keys and state key
        image_keys = [key for key in sample_obs.keys() if "observation.images" in key]
        state_key = "observation.state"
        task_encoding_key = "observation.task_encoding"
        
        # Stack state: (1, state_dim) -> (1, n_obs_steps, state_dim)
        if state_key in sample_obs:
            batch[state_key] = torch.stack([obs[state_key] for obs in processed_observations], dim=1)
        
        # Stack all camera images into a single "observation.images" tensor
        if image_keys:
            # For each timestep, stack all cameras: (1, C, H, W) -> (1, num_cameras, C, H, W)
            images_per_timestep = []
            for obs in processed_observations:
                # Stack all cameras for this timestep
                cameras_stacked = torch.stack([obs[key] for key in image_keys], dim=1)  # (1, num_cameras, C, H, W)
                images_per_timestep.append(cameras_stacked)
            
            # Stack across timesteps: (1, num_cameras, C, H, W) -> (1, n_obs_steps, num_cameras, C, H, W)
            batch["observation.images"] = torch.stack(images_per_timestep, dim=1)  # (1, n_obs_steps, num_cameras, C, H, W)
        
        # Stack task_encoding: (1, task_encoding_dim) -> (1, n_obs_steps, task_encoding_dim)
        if task_encoding_key in sample_obs:
            batch[task_encoding_key] = torch.stack([obs[task_encoding_key] for obs in processed_observations], dim=1)
        
        return batch

    def _apply_segmentation_to_observations(self, obs_list: list[TimestampedObservation]) -> list[TimestampedObservation]:
        """
        Apply segmentation tracking to raw robot observations.
        
        This method processes the configured source camera and writes its SAM2
        visualization to ``segmentation_output_key``. By default the output key
        is the source camera for backwards compatibility.
        
        Args:
            obs_list: List of TimestampedObservation objects (with raw robot observations)
            
        Returns:
            List of TimestampedObservation objects with segmented images
        """
        if not self.enable_segmentation or not self.segmentation_initialized:
            return obs_list
        
        segmented_obs_list = []
        
        for index, ts_obs in enumerate(obs_list):
            # Get the camera image from raw robot observation
            robot_obs = ts_obs.observation

            if self.segmentation_camera not in robot_obs:
                # Camera not found, keep original observation
                segmented_obs_list.append(ts_obs)
                print(f"Warning: Camera '{self.segmentation_camera}' not found in observation, skipping segmentation.")
                continue

            # Segmentation itself: on failure keep the original observation and move on.
            # Each observation must be appended exactly once, otherwise the batch ends up
            # longer than n_obs_steps.
            try:
                img_rgb = robot_obs[self.segmentation_camera]
                mask_logits, masks = self.segmenter.track_frame(img_rgb)
                segmented_image_rgb = MaskVisualizer.visualize(
                    img_rgb,
                    masks,
                    mode=self.visualization_mode,
                    **self.seg_params,
                )
            except Exception as e:
                print(f"Warning: Segmentation tracking failed: {e}")
                traceback.print_exc()
                # Keep original observation on error
                segmented_obs_list.append(ts_obs)
                continue

            new_robot_obs = robot_obs.copy()
            new_robot_obs[self.segmentation_output_key] = segmented_image_rgb

            # Create new timestamped observation
            segmented_obs_list.append(TimestampedObservation(new_robot_obs, ts_obs.timestamp))

            # Side effects for the newest frame only; never let them drop an observation.
            if index != len(obs_list) - 1:
                continue
            try:
                self._save_segmentation_artifacts(
                    frame_rgb=img_rgb,
                    masks=masks,
                    masked_image_rgb=segmented_image_rgb,
                    stage="track",
                )

                # Store latest segmentation for display in main thread (only for the last frame)
                if self.show_segmentation:
                    # Convert RGB to BGR for cv2.imshow (will be displayed in main thread)
                    display_image = cv2.cvtColor(segmented_image_rgb, cv2.COLOR_RGB2BGR)
                    with self.segmentation_display_lock:
                        self.latest_segmentation_display = display_image
            except Exception as e:
                print(f"Warning: Saving/displaying segmentation failed: {e}")
                traceback.print_exc()

        return segmented_obs_list
    
    def _predict_action_chunk_new(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict action chunk using the new batched interface.
        
        Args:
            batch: Batched observations
                - "observation.state": (1, n_obs_steps, state_dim)
                - "observation.images": (1, n_obs_steps, num_cameras, C, H, W)
                - "observation.task_encoding": (1, n_obs_steps, task_encoding_dim) [optional]

        Returns:
            Action chunk tensor: (chunk_size, action_dim) or (1, chunk_size, action_dim)
        """
        with torch.no_grad():
            # Call the new predict_action_chunk_new method
            # batch is already preprocessed in _construct_observation_batch
            action_chunk_tensor, diffusion_loss = self.policy.predict_action_chunk_new(batch, noise=None)
        if diffusion_loss is not None and diffusion_loss > self.diffusion_loss_warning_thres:
            print(f"[{time.time():.3f}] Warning! Large diffusion loss: {float(diffusion_loss)} possibly out of distribution!")
        
        return action_chunk_tensor
    
    def _aggregate_action_queue(
        self,
        incoming_actions: list[TimestampedAction],
    ):
        """
        Aggregate incoming actions with existing action queue based on timestamps.
        
        Args:
            incoming_actions: List of new TimestampedAction objects to add
        """

        # Exponential moving average aggregation
        alpha = 0.8  # Weight for new action
        aggregate_fn = lambda old_action, new_action: {
            key: alpha * new_action[key] + (1 - alpha) * old_action[key]
            for key in new_action.keys()
        }
        # Overriding aggregation
        # def aggregate_fn(x1, x2):
        #         return x2
        
        # Timestamp matching tolerance (half of control period)
        timestamp_tolerance = self.dt / 2.0
        
        current_time = time.time()
        
        with self.action_queue_lock:
            # Convert existing queue to list for easier manipulation
            existing_actions_list = []
            for action in self.action_queue:
                # Only keep future actions (not too old)
                if action.expected_timestamp >= current_time - self.dt:
                    existing_actions_list.append(action)
            
            # Process incoming actions
            for new_action in incoming_actions:
                new_timestamp = new_action.expected_timestamp
                
                # Skip actions that are too old (already executed or missed)
                if new_timestamp < current_time - self.dt:
                    continue
                
                # Find if there's an existing action with similar timestamp
                matched_idx = None
                for idx, existing_action in enumerate(existing_actions_list):
                    if abs(existing_action.expected_timestamp - new_timestamp) < timestamp_tolerance:
                        matched_idx = idx
                        break
                
                if matched_idx is not None:
                    # Found a matching timestamp, aggregate actions
                    old_action = existing_actions_list[matched_idx]
                    aggregated_action_dict = aggregate_fn(old_action.action, new_action.action)
                    
                    # Replace with aggregated action
                    existing_actions_list[matched_idx] = TimestampedAction(
                        action=aggregated_action_dict,
                        expected_timestamp=new_timestamp,  # Use new timestamp
                        base_obs_timestamp=new_action.base_obs_timestamp,
                        action_index=new_action.action_index
                    )
                else:
                    # No matching timestamp, add new action
                    existing_actions_list.append(new_action)
            
            # Rebuild queue sorted by timestamp
            self.action_queue.clear()
            existing_actions_list.sort(key=lambda x: x.expected_timestamp)
            for action in existing_actions_list:
                self.action_queue.append(action)
    
    def _policy_action_chunk_to_robot_actions(self, action_chunk_tensor: torch.Tensor) -> list[Dict[str, float]]:
        """
        Convert action chunk tensor to list of robot action dictionaries.
        Postprocesses each action individually and converts to robot action format.
        
        Args:
            action_chunk_tensor: (chunk_size, action_dim) or (1, chunk_size, action_dim)
            
        Returns:
            List of robot action dictionaries
        """
        # Remove batch dimension if present
        if action_chunk_tensor.dim() == 3:
            action_chunk_tensor = action_chunk_tensor.squeeze(0)  # (chunk_size, action_dim)
        
        action_keys = list(self.robot.action_features.keys())
        action_chunk = []
        
        for i in range(action_chunk_tensor.shape[0]):
            # Get single action tensor: (action_dim,)
            action_vec = action_chunk_tensor[i].unsqueeze(0)  # (1, action_dim) for postprocessor
            
            # Postprocess individual action
            action_vec = self.postprocessor(action_vec)  # (1, action_dim)
            action_vec = action_vec.squeeze()  # (action_dim,)
            
            # Validate action dimension
            if action_vec.shape[0] != len(action_keys):
                raise ValueError(
                    f"Action dimension mismatch: policy={action_vec.shape[0]} "
                    f"!= robot={len(action_keys)}"
                )
            
            # Convert to robot action dictionary
            robot_action = {
                key: action_vec[idx].item() 
                for idx, key in enumerate(action_keys)
            }
            action_chunk.append(robot_action)
        
        return action_chunk

    def _robot_observation_to_policy_observation(self, robot_obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Convert robot observation to policy input format.
        
        Args:
            robot_obs: Raw observation from robot
            
        Returns:
            Policy observation with proper tensor format
        """
        # Convert to lerobot dataset format
        lerobot_features = hw_to_dataset_features(
            self.robot.observation_features, 
            prefix="observation", 
            use_video=False
        )
        lerobot_obs = build_dataset_frame(lerobot_features, robot_obs, prefix="observation")
        
        image_keys = [k for k in lerobot_obs if "observation.images" in k]
        state_key = "observation.state"
        
        if state_key not in lerobot_obs:
            raise KeyError(f"{state_key} not found in lerobot_obs! Keys: {list(lerobot_obs.keys())}")
        
        policy_obs = {}
        
        state_array = np.array(lerobot_obs[state_key], dtype=np.float32, copy=True)
        # !! IMPORTANT!! Convert to tensors with non_blocking=True to avoid CUDA synchronization
        policy_obs[state_key] = torch.from_numpy(state_array).to(
            device=self.device, non_blocking=True
        ).unsqueeze(0)  # (1, state_dim)
        
        for img_key in image_keys:
            img = lerobot_obs[img_key]
            img_array = np.array(img, dtype=np.float32, copy=True) / 255.0
            # !! IMPORTANT!! Convert to tensors with non_blocking=True to avoid CUDA synchronization
            img_tensor = torch.from_numpy(img_array).to(device=self.device, non_blocking=True)
            img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            policy_obs[img_key] = img_tensor.unsqueeze(0)  # (1, C, H, W)
        
        # Add task encoding if required
        if self.has_task_encoding:
            if self.task_encoding is None:
                raise RuntimeError(
                    "Policy requires task_encoding but it has not been set! "
                    "Please call set_task() before running the controller."
                )
            policy_obs["observation.task_encoding"] = torch.from_numpy(
                self.task_encoding
            ).to(device=self.device, non_blocking=True).unsqueeze(0)  # (1, task_encoding_dim)
        
        return policy_obs

    def run_episode(
        self,
        episode_timeout: float = 60.0,
        sam2_prompts: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        termination_condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        """
        Run a full episode with async control loop.
        
        Args:
            episode_timeout: Maximum episode duration in seconds
            sam2_prompts: Optional dictionary mapping object_id to (points [shape (N, 2)], labels [shape (N,)]) tuples for SAM2 initialization.
            termination_condition: Optional callable that takes current robot observation and returns True to stop the episode early.
        """
        if not self.robot.is_connected:
            print("✗ Robot is not connected. Cannot run episode.")
            return
        
        print(f"\n{'='*40}")
        print(f"Starting Episode (timeout: {episode_timeout}s)")
        print(f"{'='*40}")
        
        # Start controller
        self.start(sam2_prompts=sam2_prompts)
        
        start_time = time.time()
        iteration = 0
        
        try:
            while self.running and (time.time() - start_time) < episode_timeout:
                loop_start = time.time()
                
                # Get observation and update controller
                obs = self.robot.get_observation()
                self.update_observation(obs)

                # Check termination condition
                if termination_condition is not None and termination_condition(obs):
                    print("\n✓ Termination condition met. Stopping episode early.")
                    break
                
                # Get action from queue
                action = self.get_action()
                
                if action is not None:
                    # Send action to robot
                    self.robot.send_action(action)
                
                # Display segmentation in main thread (if available)
                if self.show_segmentation:
                    with self.segmentation_display_lock:
                        if self.latest_segmentation_display is not None:
                            cv2.imshow("Segmentation Tracking", self.latest_segmentation_display)
                            cv2.waitKey(1)
                
                # Maintain control frequency
                elapsed = time.time() - loop_start
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)
                else:
                    print(f"⚠ WARNING: Control loop overrun: (elapsed: {elapsed:.3f}s)")
                
                iteration += 1
        
        except KeyboardInterrupt:
            print("\n✗ Episode interrupted by user")
        
        except Exception as e:
            print(f"\n✗ Episode error: {e}")
            traceback.print_exc()
        
        finally:
            # Stop controller
            self.stop()

            # Destroy windows on the thread that created them (this one).
            if self.show_segmentation:
                _safe_destroy_all_windows()

            elapsed_time = time.time() - start_time
            print(f"\nEpisode completed: {elapsed_time:.1f}s")

    def _print_stats(self):
        """Print controller statistics."""
        print("\n" + "="*40)
        print("Controller Statistics")
        print("="*40)
        if self.stats["total_inferences"] > 0:
            avg_inference = self.stats["total_inference_time"] / self.stats["total_inferences"]
            print(f"Total inferences: {self.stats['total_inferences']}")
            print(f"Average inference time: {avg_inference*1000:.1f}ms")
        else:
            print("No inferences performed")
        print("="*40)

    def __del__(self):
        """Cleanup on deletion."""
        if self.running:
            self.stop()
