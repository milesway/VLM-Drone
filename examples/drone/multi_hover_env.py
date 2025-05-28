import os
import json
import torch
import numpy as np
import genesis as gs
import cv2
from scipy.spatial.distance import cdist
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

import pickle
# from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic


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
        # self.scene.add_entity(morph=gs.morphs.Plane()) # Plane environment with no obstacles
        self.scene.add_entity(morph=gs.morphs.Mesh(file="usd/warehouse_new/warehouse-01.mtl.obj", scale=1, fixed=True, collision=False))
            
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

    def fly_to_point(self, max_steps=10000, margin=0.1):
        
        os.makedirs(self.state_params_save_path, exist_ok=True)
        # bool complete[# targets] stores the state of the target points be reached (True) of not (False)
        completed = [False] * self.n_drones
        step = 0

        while not all(completed) and step < max_steps:
            step_log = []
            last_action = torch.zeros((4,))
            for i, drone in enumerate(self.drones):
                
                # Get target assigned to drone i, controller action, save state data in log
                target_idx = self.assignments[i]
                target = self.target_trajectory[target_idx]
                pos = drone.get_pos().cpu().numpy()

                # If distance between drone i and target < margin, target reached, completed[i] = True
                if np.linalg.norm(np.array(target) - pos) < margin:
                    completed[target_idx] = True
                else:
                    completed[target_idx] = False
                    obs = self._get_obs(drone, target, last_action).unsqueeze(0)
                    # action = self.policy(obs).squeeze(0).cpu().numpy()
                    action = self.policy.act_inference(obs).squeeze(0)
                    action = torch.clip(action, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
                    rpms = (1 + action * 0.8) * self.base_rpm # match hover_env scale
                    drone.set_propellels_rpm(rpms)
                    
                    last_action = action
                    
                # Save drone state, PID gains and current target point in log for LLM input                  
                state = {
                    "drone_id": i,
                    "step": step,
                    "target": list(target),
                    "position": pos.tolist(),
                    "velocity": drone.get_vel().cpu().numpy().tolist(),
                    "attitude": quat_to_xyz(drone.get_quat(), rpy=True, degrees=True).cpu().numpy().tolist(),
                    "rpms": rpms.tolist(), # NOT parsed as LLM input at inference time
                }
                step_log.append(state)

            # Save log and snapshot every step
            full_log = {
                "step": step,
                "targets": self.target_trajectory, # list of target coordinates
                "completed": completed.copy(), # track whether each target has been reached
                "drone_states": step_log # current state parameters of all the drones
            }
            with open(os.path.join(self.state_params_save_path, f"step_{step}.json"), "w") as f:
                json.dump(full_log, f, indent=2)

            # Use genesis feature to take picture with cam and save fig
            if self.snapshots:
                # rgb, _, _, _ = self.cam.render(rgb=True, depth=True, segmentation=True, normal=True)
                rgb, _, _, _ = self.cam.render(rgb=True, depth=False, segmentation=False, normal=False)
                # Convert RGB to BGR for saving
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.picture_save_path, f"snapshot_{step:04d}.png"), bgr)

            def openai_api_call(self, picture_path):
                # Use OpenAI API to analyze the picture
                pass
            
            # Use OpenAI API to analyze the picture
            # openai_api_call(os.path.join(self.picture_save_path, f"snapshot_{step:04d}.png"))
            
            #################### pre-step ####################
            # TO DO: Call LLM API to input way-points based on current drone state
            self.scene.step()
            step += 1

    def run(self):
        self.cam.start_recording()
        for i, point in enumerate(self.target_trajectory):
            self.target_markers[i].set_pos(point)

        # Execute the drones to fly to the current set of target points
        self.fly_to_point()

        self.cam.stop_recording(save_to_filename=self.video_save_path or "./videos/rl_flight.mp4")
        print(f"Video saved to: {self.video_save_path}")
