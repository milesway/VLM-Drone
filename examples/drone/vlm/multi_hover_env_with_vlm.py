import os
import json
import torch
import numpy as np
import genesis as gs
import cv2
from scipy.spatial.distance import cdist
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

import pickle
from rsl_rl.modules import ActorCritic
from api import openai_api_call
from quadcopter_controller import DronePIDController
    
    
class MultiHoverEnv:
    def __init__(self, 
                picture_save_path, # Directory to save pictures
                video_save_path, # Directory to save videos
                state_params_save_path, # Directory to save state parameters
                target_trajectory, # List of target positions
                env_name, # Name of the environment
                save_path, # Directory to save results
                ckpt, # Checkpoint for loading trained RL model
                n_drones, # Number of drones
                llm_replan_interval, # How often to call LLM (in steps)
                use_vision, # Whether to send images to LLM
                model_name, # LLM model name
                min_distance, # Minimum safe distance between drones
                target_threshold, # Distance threshold to consider target reached   
                enable_ray_tracing, # Whether to enable ray tracing
                show_viewer, # Whether to show viewer
                pid_params = None # PID parameters for PID controller mode
        ):

        self.n_drones = n_drones
        self.target_trajectory = target_trajectory
        self.use_vision = use_vision
        self.model_name = model_name
        self.picture_save_path = picture_save_path
        self.video_save_path = video_save_path
        self.state_params_save_path = state_params_save_path
        self.enable_ray_tracing = enable_ray_tracing
        self.ckpt = ckpt
        self.env_name = env_name
        self.save_path = save_path
        
        # LLM-related parameters
        self.use_llm = llm_replan_interval > 0
        self.llm_replan_interval = llm_replan_interval
        self.min_distance = min_distance
        self.target_threshold = target_threshold
        
        self.llm_step_counter = 0
        self.current_waypoints = None  # Will store LLM-generated waypoints
        
        # Collision detection and LLM calling control
        self.collision_llm_interval = 20  # Call LLM every 20 timesteps when collision detected
        self.last_collision_llm_step = -1  # Track when last collision-triggered LLM call was made
        self.collision_detected_recently = False  # Track if collision is ongoing
        
        # Conversation history for LLM/VLM
        self.conversation_history = []  # Store conversation history
        self.max_history_length = 5  # Keep last 5 interactions to avoid token limit
        
        # Conversation logging setup
        self.conversation_log_path = os.path.join(save_path, "conversation_logs")
        os.makedirs(self.conversation_log_path, exist_ok=True)
        self.full_conversation_log = []  # Store complete conversation log (no length limit)
        self.conversation_log_file = os.path.join(self.conversation_log_path, f"{env_name}_{model_name}_{'vision' if use_vision else 'text'}_conversation_log.json")
        
        # Controller parameters
        self.control_mode = "rl" if ckpt is not None else "pid"
        self.pid_params = pid_params or self._default_pid_params()
        
        self.base_rpm = 14468.429183500699
        self.min_rpm = 0.9 * self.base_rpm
        self.max_rpm = 1.5 * self.base_rpm
        self.dt = 0.01
        
        if self.use_vision:
            self.video_save_path = os.path.join(video_save_path, f"{env_name}_flight_{model_name}_vision.mp4")
        else:
            self.video_save_path = os.path.join(video_save_path, f"{env_name}_flight_{model_name}_text.mp4")
        if not self.use_llm:
            if self.control_mode == "rl":
                self.video_save_path = os.path.join(video_save_path, f"{env_name}_flight_rl.mp4")
            else:
                self.video_save_path = os.path.join(video_save_path, f"{env_name}_flight_pid.mp4")
        
        
 
        gs.init(backend=gs.gpu)
        
        if enable_ray_tracing:
            self.scene = gs.Scene(
            show_viewer=show_viewer,
            viewer_options=gs.options.ViewerOptions(
                res=(500, 500),
                camera_pos=(8.5, 0.0, 4.5),
                camera_lookat=(3.0, 0.0, 0.5),
                camera_fov=50,
            ),
            vis_options = gs.options.VisOptions(
                show_world_frame = True, # visualize the coordinate frame of `world` at its origin
                world_frame_size = 1.0, # length of the world frame in meter
                show_link_frame  = False, # do not visualize coordinate frames of entity links
                show_cameras     = False, # do not visualize mesh and frustum of the cameras added
                plane_reflection = True, # turn on plane reflection
                ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
            ),
            sim_options=gs.options.SimOptions(dt=self.dt),
            renderer=gs.renderers.RayTracer(  # type: ignore
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path="textures/indoor_bright.png",
                ),
            ),
            env_radius=15.0,
            env_euler=(0, 0, 180),
            lights=[
                {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
            ],
        ),
        )
        else:
            self.scene = gs.Scene(
                show_viewer=show_viewer,
                sim_options=gs.options.SimOptions(dt=self.dt)
            )
            

        # Load trained RL model and config if checkpoint provided
        if self.control_mode == "rl":
            try:
                # load any checkpoint in the directory under ckpt, file name could be anything end with .pt
                ckpt_path = [os.path.join(self.ckpt, f) for f in os.listdir(self.ckpt) if f.endswith(".pt")]
                ckpt_path = ckpt_path[0]
                # load cfgs.pkl in the same directory
                env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(os.path.join(self.ckpt, "cfgs.pkl"), "rb"))
                
                self.obs_scales = obs_cfg["obs_scales"]
                self.env_cfg = env_cfg
                
                self.policy = ActorCritic(
                    **train_cfg["policy"],
                    num_actor_obs=obs_cfg["num_obs"],
                    num_critic_obs=obs_cfg["num_obs"],  # assume same as actor for inference
                    num_actions=env_cfg["num_actions"]
                ).to(gs.device)

                ckpt_data = torch.load(ckpt_path, map_location=gs.device)
                self.policy.load_state_dict(ckpt_data["model_state_dict"])
                self.policy.eval()
                print(f"Loaded RL policy from checkpoint {ckpt}")
            except Exception as e:
                print(f"Failed to load RL checkpoint: {e}, falling back to PID control")
                self.control_mode = "pid"
                self.policy = None
        else:
            print("Using PID controller (no RL checkpoint provided)")
            self.policy = None
    
        self._setup_scene() # Entities initialized: Plane, red ball targets
        self._setup_drones() # Drones initialized. controller timestep == simulation timestep.
        self._setup_camera() # Adjust viewing angle as needed

        self.scene.build()

        self.assignments = self._greedy_target_assignment() # Initial assignment

    def _default_pid_params(self):
        """Default PID parameters for drone control."""
        return [
        [2.0, 0.0, 0.0],  # pos_x controller: how fast drone should move in x-direction to reach target x
        [2.0, 0.0, 0.0],  # pos_y controller: same, for y
        [2.0, 0.0, 0.0],  # pos_z controller: same, for z (altitude)

        [20.0, 0.0, 20.0],  # vel_x controller: converts x-velocity error to roll correction
        [20.0, 0.0, 20.0],  # vel_y controller: converts y-velocity error to pitch correction
        [25.0, 0.0, 20.0],  # vel_z controller: converts z-velocity error to thrust

        [10.0, 0.0, 1.0],  # roll attitude controller: stabilizes roll angle
        [10.0, 0.0, 1.0],  # pitch attitude controller: stabilizes pitch angle
        [2.0, 0.0, 0.2],   # yaw attitude controller: stabilizes yaw angle (less aggressive)
    ]


    def _clamp_rpm(self, rpm):
        """Clamp RPM values to safe range."""
        return max(self.min_rpm, min(int(rpm), self.max_rpm))


    def _setup_scene(self):
        """Initialize plane and target spheres."""
        if self.env_name == "warehouse":        
            self.scene.add_entity(morph=gs.morphs.Mesh(file="usd/digital_updated/untitled.obj", scale=1, fixed=True, collision=False, pos=(0, 0, 0)))
        elif self.env_name == "plain":
            self.scene.add_entity(morph=gs.morphs.Plane(
                pos=(0.0, 0.0, -0.5),
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ImageTexture(
                        image_path="usd/ground/Parquet_Color.png",
                    )
                ),
            )
        else:
            self.scene.add_entity(
                morph=gs.morphs.Plane(
                    pos=(0.0, 0.0, -0.5),
                ),
                surface=gs.surfaces.Aluminium(
                    ior=10.0,
                ),
            )
            
        self.target_markers = [] # Initialize red balls as targets
        for _ in self.target_trajectory:
            marker = self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.05, fixed=True, collision=False),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0))
                )
            )
            self.target_markers.append(marker)


    def _setup_drones(self):
        """Initialize drones at random start positions."""
        self.drones = []
        self.controllers = []  # PID controllers for PID mode
        
        for _ in range(self.n_drones):
            pos = np.random.uniform(low=[-1, -1, 0.2], high=[1, 1, 0.5]) # Drone initial position
            drone = self.scene.add_entity(
                morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=pos)
            )
            self.drones.append(drone)
            
            # Create PID controller if in PID mode
            if self.control_mode == "pid":
                controller = DronePIDController(
                    drone=drone, 
                    dt=self.dt, 
                    base_rpm=self.base_rpm, 
                    pid_params=self.pid_params
                )
                self.controllers.append(controller)

    def _setup_camera(self):
        self.cam = self.scene.add_camera(
            res=(640, 480), # camera resolution 
            pos=(5.0, 1.0, 1.5),
            lookat=(0.0, 0.0, 0.5),
            fov=30,
            GUI=True, # Display rendered image from the camera when running simulation
        )
        

    def _greedy_target_assignment(self):
        """Assign each drone to the nearest available target."""
        
         # Calculate pairwise Euclidean distances between all drones and targets
        drone_pos = np.array([d.get_pos().cpu().numpy() for d in self.drones])
        target_pos = np.array(self.target_trajectory)
        distances = cdist(drone_pos, target_pos) # distances[i][j] = distance between drone i and target j

        assignment = [-1] * self.n_drones
        taken = set()
        for i in range(self.n_drones):
            # Choose nearest untaken target
            for j in np.argsort(distances[i]):
                if j not in taken:
                    assignment[i] = j
                    taken.add(j)
                    break
        return assignment

    def _prepare_vlm_input(self, step, drone_states, completed):
        """Prepare input data for VLM API call."""
        # Check for collision risks
        collision_detected = self._check_drone_collisions()
        collision_pairs = []
        
        if collision_detected:
            drone_positions = []
            for drone in self.drones:
                pos = drone.get_pos().cpu().numpy()
                drone_positions.append(pos)
            
            drone_positions = np.array(drone_positions)
            
            # Find all pairs within min_distance
            for i in range(len(drone_positions)):
                for j in range(i + 1, len(drone_positions)):
                    distance = np.linalg.norm(drone_positions[i] - drone_positions[j])
                    if distance < self.min_distance:
                        collision_pairs.append({
                            "drone_1": i,
                            "drone_2": j,
                            "distance": float(distance)
                        })
        
        # Add information about which drones have completed their targets
        completed_drones = []
        for i, drone in enumerate(self.drones):
            pos = drone.get_pos().cpu().numpy()
            if i < len(self.assignments) and self.assignments[i] >= 0:
                target_pos = np.array(self.target_trajectory[self.assignments[i]])
                distance_to_target = np.linalg.norm(pos - target_pos)
                is_completed = distance_to_target < self.target_threshold
                completed_drones.append({
                    "drone_id": i,
                    "assigned_target_id": self.assignments[i],
                    "distance_to_target": float(distance_to_target),
                    "is_completed": bool(is_completed)
                })
            else:
                completed_drones.append({
                    "drone_id": i,
                    "assigned_target_id": -1,
                    "distance_to_target": float('inf'),
                    "is_completed": False
                })
        
        vlm_input = {
            "step": step,
            "targets": self.target_trajectory,
            "completed": completed,
            "drone_states": drone_states,
            "completed_drones": completed_drones,
            "min_distance": self.min_distance,
            "target_threshold": self.target_threshold,
            "n_drones": self.n_drones,
            "collision_detected": bool(collision_detected),
            "collision_pairs": collision_pairs
        }
        return vlm_input

    def _call_vlm_for_waypoints(self, step, drone_states, completed, picture_path=None):
        """Call LLM to get new waypoints and assignments with conversation history."""
        import time
        
        if not self.use_llm:
            return None
            
        vlm_input = self._prepare_vlm_input(step, drone_states, completed)
        
        # Convert to JSON-serializable format before API call
        vlm_input_serializable = self._convert_to_json_serializable(vlm_input)
        
        # Only pass picture_path if use_vision is True
        image_path = picture_path if self.use_vision else None
        
        # Time the API call
        start_time = time.time()
        
        # Include conversation history in the API call
        try:
            response = openai_api_call(image_path, vlm_input_serializable, self.model_name, self.conversation_history, self.llm_replan_interval)
            end_time = time.time()
            response_time = end_time - start_time
            
            if response and "waypoints" in response and "assignments" in response:
                print(f"Step {step}: LLM Response - {response.get('reasoning', 'No reasoning provided')}")
                
                # Log successful conversation
                llm_call_reason = self._get_llm_call_reason(step)
                self._log_conversation(step, vlm_input_serializable, response, image_path, llm_call_reason, response_time)
                
                # Add to conversation history (use serializable version for consistency)
                self._add_to_conversation_history(step, vlm_input_serializable, response, image_path)
                
                return response
            else:
                print(f"Step {step}: Invalid LLM response, using fallback")
                
                # Log failed conversation
                llm_call_reason = self._get_llm_call_reason(step)
                error_msg = f"Invalid response format: missing waypoints or assignments. Response: {response}"
                self._log_conversation(step, vlm_input_serializable, None, image_path, llm_call_reason, response_time, error_msg)
                
                # Still add failed attempt to history for context (use serializable version)
                self._add_to_conversation_history(step, vlm_input_serializable, None, image_path)
                
                return None
                
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"Step {step}: API call failed: {str(e)}")
            
            # Log failed conversation with exception
            llm_call_reason = self._get_llm_call_reason(step)
            self._log_conversation(step, vlm_input_serializable, None, image_path, llm_call_reason, response_time, str(e))
            
            # Still add failed attempt to history for context
            self._add_to_conversation_history(step, vlm_input_serializable, None, image_path)
            
            return None

    def _get_llm_call_reason(self, step):
        """Helper to determine the reason for LLM call (for logging)."""
        collision_detected = self._check_drone_collisions()
        collision_llm_needed = (collision_detected and 
                               (step - self.last_collision_llm_step >= self.collision_llm_interval or 
                                self.last_collision_llm_step == -1))
        
        if collision_llm_needed:
            return f"collision risk detected (interval: {self.collision_llm_interval} steps)"
        elif step % self.llm_replan_interval == 0:
            return "scheduled interval"
        else:
            return "planning steps expired or no current plan"

    def _add_to_conversation_history(self, step, vlm_input, response, image_path=None):
        """Add interaction to conversation history."""
        interaction = {
            "step": step,
            "input": vlm_input,
            "response": response,
            "has_image": image_path is not None,
            "timestamp": step  # Use step as timestamp for simplicity
        }
        
        self.conversation_history.append(interaction)
        
        # Keep only the last max_history_length interactions
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def _get_conversation_context(self):
        """Get formatted conversation history for LLM context."""
        if not self.conversation_history:
            return ""
        
        context = "\n\nPrevious interactions:\n"
        for i, interaction in enumerate(self.conversation_history[:-1]):  # Exclude current interaction
            context += f"\nStep {interaction['step']}:\n"
            context += f"Input: {json.dumps(interaction['input'], indent=2)}\n"
            if interaction['response']:
                context += f"Response: {json.dumps(interaction['response'], indent=2)}\n"
            else:
                context += "Response: Failed to generate valid response\n"
            context += f"Had image: {interaction['has_image']}\n"
        
        return context

    def _get_obs(self, drone, target, last_action):
        """Get observation for RL policy. Only used in RL mode."""
        if self.control_mode != "rl" or self.policy is None:
            return None
            
        pos = drone.get_pos()
        vel = drone.get_vel()
        quat = drone.get_quat()
        ang = drone.get_ang()

        rel_pos = torch.tensor(target, device=gs.device) - pos
        base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(quat) * inv_quat(quat), quat), rpy=True, degrees=True
        )
        lin_vel = transform_by_quat(vel, inv_quat(quat))
        ang_vel = transform_by_quat(ang, inv_quat(quat))

        obs = torch.cat([
            torch.clip(rel_pos * self.obs_scales["rel_pos"], -1, 1),
            quat,
            torch.clip(lin_vel * self.obs_scales["lin_vel"], -1, 1),
            torch.clip(ang_vel * self.obs_scales["ang_vel"], -1, 1),
            last_action  # placeholder for last action
        ])
        return obs

    def _convert_to_json_serializable(self, obj):
        """Convert numpy/torch types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):  # Handle numpy boolean type
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        else:
            return obj

    def _check_drone_collisions(self):
        """Check if any pair of drones is within min_distance of each other."""
        if not self.use_llm:
            return False
            
        drone_positions = []
        for drone in self.drones:
            pos = drone.get_pos().cpu().numpy()
            drone_positions.append(pos)
        
        drone_positions = np.array(drone_positions)
        
        # Calculate pairwise distances between all drones
        for i in range(len(drone_positions)):
            for j in range(i + 1, len(drone_positions)):
                distance = np.linalg.norm(drone_positions[i] - drone_positions[j])
                if distance < self.min_distance:
                    print(f"⚠️  Collision risk detected: Drones {i} and {j} are {distance:.3f}m apart (min: {self.min_distance}m)")
                    return True
        return False

    def step(self, max_steps=10000, margin=0.1):
        
        completed = [False] * len(self.target_trajectory)
        step = 0
        llm_steps_remaining = 0
        current_llm_response = None
        
        # Track collision statistics
        collision_count = 0
        collision_triggered_calls = 0

        while not all(completed) and step < max_steps:
            step_log = []
            
            # Collect current drone states for LLM input
            for i, drone in enumerate(self.drones):
                pos = drone.get_pos().cpu().numpy()
                
                # Check if any target is reached
                for target_idx, target in enumerate(self.target_trajectory):
                    if np.linalg.norm(np.array(target) - pos) < margin:
                        completed[target_idx] = True
                
                # Save drone state for LLM input                  
                state = {
                    "drone_id": i,
                    "step": step,
                    "position": pos.tolist(),
                    "velocity": drone.get_vel().cpu().numpy().tolist(),
                    "attitude": quat_to_xyz(drone.get_quat(), rpy=True, degrees=True).cpu().numpy().tolist(),
                }
                step_log.append(state)

            # Take snapshot for LLM and/or saving
            picture_path = None
            if self.picture_save_path:
                rgb, _, _, _ = self.cam.render(rgb=True, depth=False, segmentation=False, normal=False)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                picture_path = os.path.join(self.picture_save_path, f"snapshot_{step:04d}.png")
                cv2.imwrite(picture_path, bgr)

            # Check for collision risk
            collision_detected = self._check_drone_collisions()
            if collision_detected:
                collision_count += 1
                self.collision_detected_recently = True
            else:
                self.collision_detected_recently = False
            
            # Call LLM every llm_replan_interval steps, if no current plan, or if collision detected (with interval)
            should_call_llm = False
            llm_call_reason = ""
            
            if self.use_llm:
                # Check if we should call LLM due to collision (with 20-timestep interval)
                collision_llm_needed = (collision_detected and 
                                       (step - self.last_collision_llm_step >= self.collision_llm_interval or 
                                        self.last_collision_llm_step == -1))
                
                if collision_llm_needed:
                    should_call_llm = True
                    llm_call_reason = "collision risk detected"
                    collision_triggered_calls += 1
                    self.last_collision_llm_step = step
                elif collision_detected and step - self.last_collision_llm_step < self.collision_llm_interval:
                    # Collision detected but within interval - skip LLM call
                    steps_until_next_collision_call = self.collision_llm_interval - (step - self.last_collision_llm_step)
                    print(f"⏳ Step {step}: Collision detected but skipping LLM call (next collision call in {steps_until_next_collision_call} steps)")
                elif step % self.llm_replan_interval == 0:
                    should_call_llm = True
                    llm_call_reason = "scheduled interval"
                elif llm_steps_remaining <= 0:
                    should_call_llm = True
                    llm_call_reason = "planning steps expired"
                elif current_llm_response is None:
                    should_call_llm = True
                    llm_call_reason = "no current plan"
            
            if should_call_llm:
                print("Current LLM input:")
                print(self._prepare_vlm_input(step, step_log, completed))
                print(f"Step {step}: Calling LLM for waypoint planning ({llm_call_reason})...")
                current_llm_response = self._call_vlm_for_waypoints(step, step_log, completed, picture_path)
                
                if current_llm_response:
                    print(f"Step {step}: LLM Response - {current_llm_response.get('reasoning', 'No reasoning provided')}")
                    print(f"Step {step}: LLM Response - {current_llm_response.get('waypoints', 'No waypoints provided')}")
                    print(f"Step {step}: LLM Response - {current_llm_response.get('assignments', 'No assignments provided')}")
                    print(f"Step {step}: LLM Response - {current_llm_response.get('n_steps', 'No n_steps provided')}")
                    llm_steps_remaining = current_llm_response.get("n_steps", self.llm_replan_interval)
                    # Update assignments from LLM
                    if "assignments" in current_llm_response:
                        new_assignments = [-1] * self.n_drones
                        for assignment in current_llm_response["assignments"]:
                            drone_id = assignment["drone_id"]
                            target_id = assignment["target_id"]
                            if 0 <= drone_id < self.n_drones and 0 <= target_id < len(self.target_trajectory):
                                new_assignments[drone_id] = target_id
                        self.assignments = new_assignments
                else:
                    # Fallback to greedy assignment if LLM fails
                    print(f"Step {step}: Using fallback greedy assignment")
                    self.assignments = self._greedy_target_assignment()
                    llm_steps_remaining = self.llm_replan_interval
            elif not self.use_llm and step == 0:
                # For non-LLM mode, just use greedy assignment once at the start
                print(f"Step {step}: LLM disabled, using greedy assignment")
                self.assignments = self._greedy_target_assignment()

            # Execute drone actions
            last_actions = []
            for i, drone in enumerate(self.drones):
                pos = drone.get_pos().cpu().numpy()
                
                # Check if drone has completed its target
                drone_completed = False
                if i < len(self.assignments) and self.assignments[i] >= 0:
                    target_pos = np.array(self.target_trajectory[self.assignments[i]])
                    distance_to_target = np.linalg.norm(pos - target_pos)
                    drone_completed = distance_to_target < self.target_threshold
                
                # Determine target: use LLM waypoint if available, otherwise assigned target
                if (current_llm_response and "waypoints" in current_llm_response):
                    # Find waypoint for this drone
                    target = None
                    for waypoint_data in current_llm_response["waypoints"]:
                        if waypoint_data["drone_id"] == i:
                            target = waypoint_data["waypoint"]
                            break
                    
                    # Fallback logic based on drone completion status
                    if target is None:
                        if drone_completed and i < len(self.assignments) and self.assignments[i] >= 0:
                            # Completed drone without waypoint: hover at target
                            target = self.target_trajectory[self.assignments[i]]
                            print(f"Step {step}: Completed drone {i} missing waypoint, hovering at target")
                        elif i < len(self.assignments) and self.assignments[i] >= 0:
                            # Active drone without waypoint: move to assigned target
                            target = self.target_trajectory[self.assignments[i]]
                            print(f"Step {step}: Active drone {i} missing waypoint, using assigned target")
                        else:
                            # Unassigned drone: hover in place
                            target = pos.tolist()
                            print(f"Step {step}: Unassigned drone {i} missing waypoint, hovering in place")
                else:
                    # No LLM response: use assigned target or hover
                    if i < len(self.assignments) and self.assignments[i] >= 0:
                        target = self.target_trajectory[self.assignments[i]]
                    else:
                        print(f"Step {step}: No target for drone {i}, hovering in place")
                        target = pos.tolist()  # Hover in place

                # Control drone towards target
                rpms = [self.base_rpm] * 4  # Default hover RPMs
                if target is not None:  # Always control towards target if we have one
                    if self.control_mode == "pid":
                        # Use PID controller
                        controller = self.controllers[i]
                        rpms_raw = controller.update(target)
                        rpms = [self._clamp_rpm(rpm) for rpm in rpms_raw]
                    elif self.control_mode == "rl" and self.policy is not None:
                        # Use RL policy
                        last_action = torch.zeros((4,))
                        obs = self._get_obs(drone, target, last_action)
                        if obs is not None:
                            obs = obs.unsqueeze(0)
                            action = self.policy.act_inference(obs).squeeze(0)
                            action = torch.clip(action, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
                            rpms_tensor = (1 + action * 0.8) * self.base_rpm
                            rpms = [self._clamp_rpm(rpm.item()) for rpm in rpms_tensor]
                            last_actions.append(action)
                
                # Set propeller RPMs
                drone.set_propellels_rpm(rpms)
                    
                # Update state log with target and control info
                step_log[i].update({
                    "target": target,
                    "control_mode": self.control_mode,
                    "rpms": rpms,
                    "drone_completed": drone_completed,
                    "distance_to_target": distance_to_target if 'distance_to_target' in locals() else None,
                })

            # Save full log
            full_log = {
                "step": step,
                "targets": self.target_trajectory,
                "completed": completed.copy(),
                "drone_states": step_log,
                "vlm_response": current_llm_response,
                "assignments": self.assignments,
                "collision_detected": collision_detected,
                "llm_call_reason": llm_call_reason if should_call_llm else None,
                "conversation_history": self.conversation_history.copy() if self.conversation_history else [],
                "collision_count": collision_count,
                "collision_triggered_calls": collision_triggered_calls
            }
            
            if self.state_params_save_path:
                with open(os.path.join(self.state_params_save_path, f"step_{step}.json"), "w") as f:
                    json.dump(self._convert_to_json_serializable(full_log), f, indent=2)

            # Step simulation
            self.scene.step()
            step += 1
            llm_steps_remaining -= 1

        print(f"Simulation completed after {step} steps")
        print(f"Targets completed: {sum(completed)}/{len(completed)}")
        
        # Return statistics for run() method summary
        return {
            "total_steps": step,
            "targets_completed": sum(completed),
            "total_targets": len(completed),
            "collision_count": collision_count,
            "collision_triggered_calls": collision_triggered_calls
        }

    def run(self):
        """Run the simulation with LLM-guided waypoint planning."""
        self.cam.start_recording()
        
        # Position target markers
        for i, point in enumerate(self.target_trajectory):
            self.target_markers[i].set_pos(point)

        print(f"Starting simulation with {self.control_mode} control mode")
        print(f"LLM enabled: {self.use_llm}, Vision enabled: {self.use_vision}")
        print(f"Model: {self.model_name}")
        print(f"LLM replan interval: {self.llm_replan_interval} steps")
        print(f"Collision detection LLM interval: {self.collision_llm_interval} steps")
        print(f"Conversation history tracking: {'Enabled' if self.use_llm else 'Disabled'}")
        print(f"Max history length: {self.max_history_length if self.use_llm else 'N/A'}")
        
        # Execute the drones with LLM-guided waypoint planning
        stats = self.step()

        # Print conversation history summary
        if self.use_llm and self.conversation_history:
            print(f"\n📊 Conversation History Summary:")
            print(f"Total interactions: {len(self.conversation_history)}")
            successful_interactions = sum(1 for h in self.conversation_history if h['response'] is not None)
            print(f"Successful responses: {successful_interactions}/{len(self.conversation_history)}")
            vision_interactions = sum(1 for h in self.conversation_history if h['has_image'])
            print(f"Vision-enabled interactions: {vision_interactions}/{len(self.conversation_history)}")
            
            # Add collision statistics
            print(f"🚨 Collision Detection Summary:")
            print(f"Total collision risks detected: {stats['collision_count']}")
            print(f"LLM calls triggered by collisions: {stats['collision_triggered_calls']}")
            if stats['collision_count'] > 0:
                print(f"Collision avoidance success rate: {((stats['collision_count'] - stats['collision_triggered_calls']) / stats['collision_count'] * 100):.1f}%")
            else:
                print(f"No collisions detected during simulation ✅")

        self.cam.stop_recording(save_to_filename=self.video_save_path or "./videos/llm_flight.mp4")
        print(f"Video saved to: {self.video_save_path or './videos/llm_flight.mp4'}")
        
        # Save final conversation summary
        if self.use_llm:
            self._save_conversation_summary()
            print(f"📋 Complete conversation log available at: {self.conversation_log_file}")
            print(f"📊 Conversation summary saved for analysis")

    def _log_conversation(self, step, vlm_input, response, image_path=None, llm_call_reason="", response_time=None, error_message=None):
        """Log detailed conversation data to both memory and file."""
        import time
        
        # Create detailed log entry
        log_entry = {
            "step": step,
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "llm_call_reason": llm_call_reason,
            "input_data": vlm_input,
            "response": response,
            "has_image": image_path is not None,
            "image_path": image_path if image_path else None,
            "model_name": self.model_name,
            "use_vision": self.use_vision,
            "response_time_seconds": response_time,
            "success": response is not None,
            "error_message": error_message,
            "drone_count": self.n_drones,
            "target_count": len(self.target_trajectory),
            "collision_detected": vlm_input.get("collision_detected", False) if vlm_input else False,
            "collision_pairs": vlm_input.get("collision_pairs", []) if vlm_input else [],
        }
        
        # Add response analysis if successful
        if response:
            log_entry["response_analysis"] = {
                "has_assignments": "assignments" in response,
                "has_waypoints": "waypoints" in response,
                "has_reasoning": "reasoning" in response,
                "has_n_steps": "n_steps" in response,
                "num_assignments": len(response.get("assignments", [])),
                "num_waypoints": len(response.get("waypoints", [])),
                "n_steps_value": response.get("n_steps", None),
                "reasoning_length": len(response.get("reasoning", "")) if response.get("reasoning") else 0
            }
        
        # Add to full conversation log (no length limit)
        self.full_conversation_log.append(log_entry)
        
        # Save to file immediately (for real-time logging)
        self._save_conversation_log()
        
        print(f"📝 Logged conversation step {step} to {self.conversation_log_file}")

    def _save_conversation_log(self):
        """Save the complete conversation log to file."""
        try:
            with open(self.conversation_log_file, 'w') as f:
                json.dump(self.full_conversation_log, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Failed to save conversation log: {e}")

    def _save_conversation_summary(self):
        """Save a summary of the conversation at the end of simulation."""
        if not self.full_conversation_log:
            return
            
        summary = {
            "simulation_info": {
                "env_name": self.env_name,
                "model_name": self.model_name,
                "use_vision": self.use_vision,
                "use_llm": self.use_llm,
                "control_mode": self.control_mode,
                "n_drones": self.n_drones,
                "n_targets": len(self.target_trajectory),
                "llm_replan_interval": self.llm_replan_interval,
                "min_distance": self.min_distance,
                "target_threshold": self.target_threshold
            },
            "conversation_stats": {
                "total_interactions": len(self.full_conversation_log),
                "successful_responses": sum(1 for entry in self.full_conversation_log if entry["success"]),
                "failed_responses": sum(1 for entry in self.full_conversation_log if not entry["success"]),
                "vision_enabled_calls": sum(1 for entry in self.full_conversation_log if entry["has_image"]),
                "collision_triggered_calls": sum(1 for entry in self.full_conversation_log if "collision" in entry["llm_call_reason"]),
                "scheduled_calls": sum(1 for entry in self.full_conversation_log if "interval" in entry["llm_call_reason"]),
                "avg_response_time": sum(entry["response_time_seconds"] or 0 for entry in self.full_conversation_log) / len(self.full_conversation_log) if self.full_conversation_log else 0
            },
            "error_analysis": [
                {
                    "step": entry["step"],
                    "error": entry["error_message"],
                    "reason": entry["llm_call_reason"]
                }
                for entry in self.full_conversation_log 
                if entry["error_message"]
            ],
            "response_analysis": {
                "successful_responses": [
                    {
                        "step": entry["step"],
                        "reason": entry["llm_call_reason"],
                        "analysis": entry.get("response_analysis", {})
                    }
                    for entry in self.full_conversation_log 
                    if entry["success"]
                ]
            }
        }
        
        summary_file = os.path.join(self.conversation_log_path, f"{self.env_name}_{self.model_name}_{'vision' if self.use_vision else 'text'}_summary.json")
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"📊 Conversation summary saved to {summary_file}")
        except Exception as e:
            print(f"⚠️  Failed to save conversation summary: {e}")
        
