import os
import json
import torch
import numpy as np
import genesis as gs
import cv2
import base64
from scipy.spatial.distance import cdist
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

import pickle
# from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
from testLLM import openai_api_call


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class MultiHoverEnv:
    def __init__(self, target_trajectory,
                 n_drones=1,
                 show_viewer=True,
                 snapshots=True,
                 snap_interval=100,
                 picture_save_path=None,
                 video_save_path=None,
                 state_params_save_path=None,
                 rl_checkpoint_path=None, # Directory for accessing RL config file
                 ckpt=None, # Checkpoint for loading trained RL model
                 use_vlm=True, # Whether to use VLM for waypoint planning
                 vlm_replan_interval=1, # How often to call VLM (in steps)
                 min_distance=0.3, # Minimum safe distance between drones
                 target_threshold=0.1, # Distance threshold to consider target reached
                #  policy=None,
                #  device=None,
        ):

        self.n_drones = n_drones
        self.target_trajectory = target_trajectory
        self.snapshots = snapshots
        self.snap_interval = snap_interval
        self.picture_save_path = picture_save_path
        self.video_save_path = video_save_path
        self.state_params_save_path = state_params_save_path
        
        self.rl_checkpoint_path = rl_checkpoint_path
        self.ckpt = ckpt
        
        # VLM-related parameters
        self.use_vlm = use_vlm
        self.vlm_replan_interval = vlm_replan_interval
        self.min_distance = min_distance
        self.target_threshold = target_threshold
        self.vlm_step_counter = 0
        self.current_waypoints = None  # Will store VLM-generated waypoints
        # self.policy = policy
        # self.device = device or torch.device("cpu")

        self.base_rpm = 14468.429183500699
        self.dt = 0.01

        gs.init(backend=gs.gpu)
        self.scene = gs.Scene(
            show_viewer=show_viewer,
            sim_options=gs.options.SimOptions(dt=self.dt)
        )

        # Load trained RL model and config
        ckpt_path = os.path.join(rl_checkpoint_path, f"model_{ckpt}.pt")
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(os.path.join(rl_checkpoint_path, "cfgs.pkl"), "rb"))
        
        self.obs_scales = obs_cfg["obs_scales"]
        self.env_cfg = env_cfg
        # runner = OnPolicyRunner(None, train_cfg, rl_checkpoint_path, device=gs.device)
        # runner.load(ckpt_path)
        # policy = runner.get_inference_policy(device=gs.device)
        # policy.eval()
        
        policy = ActorCritic(
            **train_cfg["policy"],
            num_actor_obs=obs_cfg["num_obs"],
            num_critic_obs=obs_cfg["num_obs"],  # assume same as actor for inference
            num_actions=env_cfg["num_actions"]
        ).to(gs.device)

        ckpt_data = torch.load(ckpt_path, map_location=gs.device)
        policy.load_state_dict(ckpt_data["model_state_dict"])

        policy.eval()
        self.policy = policy
    
        self._setup_scene() # Entities initialized: Plane, red ball targets. TO DO: Use more complex scenes (including obstacles) as simulation environment?
        self._setup_drones() # Drones initialized. controller timestep == simulation timestep.
        self._setup_camera() # Adjust viewing angle as needed. TO DO: camera on drone?

        self.scene.build()

        self.assignments = self._greedy_target_assignment() # Assign each drone to the nearest target. TO BE replaced by LLM assignments@!

    def _setup_scene(self):
        """Initialize plane and target spheres."""
        self.scene.add_entity(morph=gs.morphs.Plane()) # Plane environment with no obstacles
        # self.scene.add_entity(morph=gs.morphs.Mesh(file="/usd/Collected_simple_room/simple_room.usd", scale=1, fixed=True, collision=True))
            
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
        for _ in range(self.n_drones):
            pos = np.random.uniform(low=[-1, -1, 0.2], high=[1, 1, 0.5]) # Drone initial position
            drone = self.scene.add_entity(
                morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=pos)
            )
            self.drones.append(drone)

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
        vlm_input = {
            "step": step,
            "targets": self.target_trajectory,
            "completed": completed,
            "drone_states": drone_states,
            "min_distance": self.min_distance,
            "target_threshold": self.target_threshold,
            "n_drones": self.n_drones
        }
        return vlm_input

    def _call_vlm_for_waypoints(self, step, drone_states, completed, picture_path=None):
        """Call VLM to get new waypoints and assignments."""
        if not self.use_vlm:
            return None
            
        try:
            vlm_input = self._prepare_vlm_input(step, drone_states, completed)
            response = openai_api_call(picture_path, vlm_input)
            
            if response and "waypoints" in response and "assignments" in response:
                print(f"Step {step}: VLM Response - {response.get('reasoning', 'No reasoning provided')}")
                return response
            else:
                print(f"Step {step}: Invalid VLM response, using fallback")
                return None
                
        except Exception as e:
            print(f"Step {step}: VLM call failed: {e}, using fallback")
            return None

    def _get_obs(self, drone, target, last_action):
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

    def step(self, max_steps=10000, margin=0.1):
        
        completed = [False] * len(self.target_trajectory)
        step = 0
        vlm_steps_remaining = 0
        current_vlm_response = None

        while not all(completed) and step < max_steps:
            step_log = []
            
            # Collect current drone states for VLM input
            for i, drone in enumerate(self.drones):
                pos = drone.get_pos().cpu().numpy()
                
                # Check if any target is reached
                for target_idx, target in enumerate(self.target_trajectory):
                    if np.linalg.norm(np.array(target) - pos) < margin:
                        completed[target_idx] = True
                
                # Save drone state for VLM input                  
                state = {
                    "drone_id": i,
                    "step": step,
                    "position": pos.tolist(),
                    "velocity": drone.get_vel().cpu().numpy().tolist(),
                    "attitude": quat_to_xyz(drone.get_quat(), rpy=True, degrees=True).cpu().numpy().tolist(),
                }
                step_log.append(state)

            # Take snapshot for VLM
            picture_path = None
            if self.snapshots:
                rgb = self.cam.render(rgb=True)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                picture_path = os.path.join(self.picture_save_path, f"snapshot_{step:04d}.png")
                cv2.imwrite(picture_path, bgr)

            # Call VLM every vlm_replan_interval steps or if no current plan
            if (step % self.vlm_replan_interval == 0 or vlm_steps_remaining <= 0 or current_vlm_response is None):
                print(f"Step {step}: Calling VLM for waypoint planning...")
                current_vlm_response = self._call_vlm_for_waypoints(step, step_log, completed, picture_path)
                
                if current_vlm_response:
                    vlm_steps_remaining = current_vlm_response.get("n_steps", self.vlm_replan_interval)
                    # Update assignments from VLM
                    if "assignments" in current_vlm_response:
                        new_assignments = [-1] * self.n_drones
                        for assignment in current_vlm_response["assignments"]:
                            drone_id = assignment["drone_id"]
                            target_id = assignment["target_id"]
                            if 0 <= drone_id < self.n_drones and 0 <= target_id < len(self.target_trajectory):
                                new_assignments[drone_id] = target_id
                        self.assignments = new_assignments
                else:
                    # Fallback to greedy assignment if VLM fails
                    print(f"Step {step}: Using fallback greedy assignment")
                    self.assignments = self._greedy_target_assignment()
                    vlm_steps_remaining = self.vlm_replan_interval

            # Execute drone actions
            last_actions = []
            for i, drone in enumerate(self.drones):
                pos = drone.get_pos().cpu().numpy()
                
                # Determine target: use VLM waypoint if available, otherwise assigned target
                if (current_vlm_response and "waypoints" in current_vlm_response):
                    # Find waypoint for this drone
                    target = None
                    for waypoint_data in current_vlm_response["waypoints"]:
                        if waypoint_data["drone_id"] == i:
                            target = waypoint_data["waypoint"]
                            break
                    
                    # Fallback to assigned target if no waypoint found
                    if target is None and i < len(self.assignments) and self.assignments[i] >= 0:
                        target = self.target_trajectory[self.assignments[i]]
                        print(f"Step {step}: No VLM waypoint for drone {i}, using assigned target")
                else:
                    # Use assigned target
                    if i < len(self.assignments) and self.assignments[i] >= 0:
                        target = self.target_trajectory[self.assignments[i]]
                    else:
                        print(f"Step {step}: No target for drone {i}, hovering in place")
                        target = pos.tolist()  # Hover in place

                # Control drone towards target
                if target is not None:
                    last_action = torch.zeros((4,))
                    if not np.allclose(pos, target, atol=margin):
                        obs = self._get_obs(drone, target, last_action).unsqueeze(0)
                        action = self.policy.act_inference(obs).squeeze(0)
                        action = torch.clip(action, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
                        rpms = (1 + action * 0.8) * self.base_rpm
                        drone.set_propellels_rpm(rpms)
                        last_action = action
                    
                    # Update state log with target and control info
                    step_log[i].update({
                        "target": target,
                        "rpms": rpms.tolist() if 'rpms' in locals() else [self.base_rpm] * 4,
                    })
                    last_actions.append(last_action)

            # Save full log
            full_log = {
                "step": step,
                "targets": self.target_trajectory,
                "completed": completed.copy(),
                "drone_states": step_log,
                "vlm_response": current_vlm_response,
                "assignments": self.assignments
            }
            
            if self.state_params_save_path:
                with open(os.path.join(self.state_params_save_path, f"step_{step}.json"), "w") as f:
                    json.dump(full_log, f, indent=2)

            # Step simulation
            self.scene.step()
            step += 1
            vlm_steps_remaining -= 1

        print(f"Simulation completed after {step} steps")
        print(f"Targets completed: {sum(completed)}/{len(completed)}")

    def run(self):
        self.cam.start_recording()
        
        # Position target markers
        for i, point in enumerate(self.target_trajectory):
            self.target_markers[i].set_pos(point)

        # Execute the drones with VLM-guided waypoint planning
        self.step()

        self.cam.stop_recording(save_to_filename=self.video_save_path or "./videos/vlm_flight.mp4")
        print(f"Video saved to: {self.video_save_path or './videos/vlm_flight.mp4'}")
