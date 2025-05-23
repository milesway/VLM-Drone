import os
import json
import numpy as np
import genesis as gs
import cv2
from scipy.spatial.distance import cdist
from genesis.engine.entities.drone_entity import DroneEntity
from quadcopter_controller import DronePIDController

import torch
import pickle
from rsl_rl.runners import OnPolicyRunner
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


class RLController:
    def __init__(self, policy, env, device):
        self.policy = policy
        self.env = env
        self.device = device

    def update(self, obs_tensor):
        with torch.no_grad():
            actions = self.policy(obs_tensor.to(self.device))
        return actions
    
class PidEnv:
    def __init__(self, target_trajectory, 
                 pid_params=None,
                 n_drones=1, 
                 show_viewer=True, 
                 snapshots=True, 
                 snap_interval=100,
                 picture_save_path=None, 
                 video_save_path=None, 
                 state_params_save_path=None,
                 
                 rl_pkl_path="logs/drone-demo",
                 control_mode="pid", 
                #  policy=None, 
                #  runner_cfg=None, 
                #  device=None
        ):

        self.base_rpm = 14468.429183500699
        self.min_rpm = 0.9 * self.base_rpm
        self.max_rpm = 1.5 * self.base_rpm
        self.pid_params = pid_params or self.default_pid_params()
        self.n_drones = n_drones
        self.target_trajectory = target_trajectory
        # self.record = record
        self.snapshots = snapshots
        self.snap_interval = snap_interval
        
        # Include option to use trained RL policy as "controller"
        self.control_mode = control_mode
        # self.policy = policy
        # self.device = device
        
        # if control_mode == "rl":
        #     assert policy is not None, "RL mode requires a trained policy."
        #     self.rl_controller = RLController(policy=policy, env=self, device=device)

        # For saving pictures and videos
        self.picture_save_path = picture_save_path
        self.video_save_path = video_save_path
        self.state_params_save_path = state_params_save_path
        gs.init(backend=gs.gpu) # Scene initialized

        self.scene = gs.Scene(
            show_viewer=show_viewer,
            sim_options=gs.options.SimOptions(dt=0.01), ###### Simulation timestep dt
        )

        if control_mode == "rl":
            # Load trained RL policy
            # log_dir = "logs/drone-demo"
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
                open(os.path.join(rl_pkl_path, "cfgs.pkl"), "rb")
            )
            runner = OnPolicyRunner(None, train_cfg, rl_pkl_path, device=gs.device)
            runner.load(os.path.join(rl_pkl_path, "model_300.pt"))
            policy = runner.get_inference_policy(device=gs.device)
            policy.eval()
            self.rl_controller = RLController(policy=policy, env=self, device=device)
        
        
        self._setup_scene() # Entities initialized: Plane, red ball targets. TO DO: Use more complex scenes (including obstacles) as simulation environment?
        self._setup_drones() # Drones initialized. controller timestep == simulation timestep. TO DO: multile drones
        self._setup_camera() # Adjust viewing angle as needed. TO DO: camera on drone?

        self.scene.build()
        self.assignments = self._greedy_target_assignment()

    def default_pid_params(self):
        return [
            [200.0, 0.0, 0.0],  # pos_x controller: how fast drone should move in x-direction to reach target x
            [200.0, 0.0, 0.0],  # pos_y controller: same, for y
            [200.0, 0.0, 0.0],  # pos_z controller: same, for z (altitude)
            [2000.0, 0.0, 2000.0],  # vel_x controller: converts x-velocity error to roll correction
            [2000.0, 0.0, 2000.0],  # vel_y controller: converts y-velocity error to pitch correction
            [10000.0, 0.0, 8000.0],  # vel_z controller: converts z-velocity error to thrust
            [100.0, 0.0, 10.0],   # roll attitude controller: stabilizes roll angle
            [100.0, 0.0, 10.0],   # pitch attitude controller: stabilizes pitch angle
            [20.0, 0.0, 2],    # yaw attitude controller: stabilizes yaw angle (less aggressive)
        ]
        
    def set_pid_params(self, new_params):
        self.controller.set_pid_params(new_params)
        
    def _setup_scene(self):
        "Initialize Assets before building scene"
        
        self.scene.add_entity(morph=gs.morphs.Plane()) # Plane environment with no obstacles
        
        self.target_markers = [] # Initialize red balls as targets
        for _ in range(len(self.target_trajectory)):
            marker = self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.05, fixed=True, collision=False),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0))
                ),
            )
            self.target_markers.append(marker)

    def _setup_drones(self):
        "Initialize drones at random start points"
        
        self.drones = []
        self.controllers = []
        for _ in range(self.n_drones):
            start_pos = np.random.uniform(low=[-1, -1, 0.2], high=[1, 1, 0.5])
            drone = self.scene.add_entity(
                morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=start_pos)
            )
            
            ###### controller timestep dt synchronized with simulation timestep dt
            if self.control_mode == "pid":
                controller = DronePIDController(drone=drone, dt=0.01, base_rpm=self.base_rpm, pid_params=self.pid_params)
                self.controllers.append(controller)
            self.drones.append(drone)
            
    def _greedy_target_assignment(self):
        "Greedy assignment of drones to nearest target pointgs"
        
        # Calculate pairwise Euclidean distances between all drones and targets
        drone_pos = np.array([d.get_pos().cpu().numpy() for d in self.drones])
        target_pos = np.array(self.target_trajectory)
        distances = cdist(drone_pos, target_pos) # distances[i][j] = distance between drone i and target j

        assignment = [-1] * self.n_drones
        taken_targets = set()

        for i in range(self.n_drones):
            # Choose nearest untaken target
            for target_idx in np.argsort(distances[i]):
                if target_idx not in taken_targets:
                    assignment[i] = target_idx
                    taken_targets.add(target_idx)
                    break

        return assignment
    
    def _setup_camera(self):
        self.cam = self.scene.add_camera(
            res=(640, 480), # camera resolution 
            pos=(5.0, 1.0, 1.5),
            lookat=(0.0, 0.0, 0.5),
            fov=30,
            GUI=True, # Display rendered image from the camera when running simulation
        )

    def clamp(self, rpm):
        return max(self.min_rpm, min(int(rpm), self.max_rpm))
    
    def fly_to_point(self, state_params_save_path=None, picture_save_path=None, max_steps=10000, margin=0.1):
       
        os.makedirs(state_params_save_path, exist_ok=True)
        # bool complete[# targets] stores the state of the target points be reached (True) of not (False)
        completed = [False] * self.n_drones 
        step = 0
        
        while not all(completed) and step < max_steps:
            step_log = []
            for i, (drone, controller) in enumerate(zip(self.drones, self.controllers)):
                
                # Get target assigned to drone i, controller action, save state data in log
                target_idx = self.assignments[i]
                target = self.target_trajectory[target_idx]
                pos = drone.get_pos().cpu().numpy()
                distance = np.linalg.norm(np.array(target) - pos)
                
                # If distance between drone i and target < margin, target reached, completed[i] = True
                if distance < margin: 
                    completed[target_idx] = True
                else:
                    completed[target_idx] = False

                if self.control_mode == "pid":
                    rpms = controller.update(target).cpu().numpy()
                elif self.control_mode == "rl":
                    obs, _ = self.rl_obs_from_state(i, target)
                    action = self.rl_controller.update(obs.unsqueeze(0)).squeeze(0).cpu().numpy()
                    rpms = (1 + action * 0.8) * self.base_rpm  # match hover_env scale

                drone.set_propellels_rpm([self.clamp(rpm) for rpm in rpms])
                
                # Save drone state, PID gains and current target point in log for LLM input
                # DO NOT parse "pid_params" and "rpms" as LLM input at inference time
                state = {
                    "drone_id": i,
                    "step": step,
                    "target": list(target), # target assined to current drone
                    "position": pos.tolist(),
                    "velocity": drone.get_vel().cpu().numpy().tolist(),
                    "attitude": controller._DronePIDController__get_drone_att().cpu().numpy().tolist(),
                    "pid_params": controller.get_pid_params(), # NOT parsed as LLM input at inference time
                    "rpms": [self.clamp(rpm) for rpm in rpms], # NOT parsed as LLM input at inference time
                }
                step_log.append(state)
                
                # Save log and snapshot every step
                full_log = {
                    "step": step, 
                    "targets": self.target_trajectory, # list of target coordinates
                    "completed": completed.copy(), # track whether each target has been reached
                    "drone_states": step_log, # current state parameters of all the drones
                }

            with open(os.path.join(state_params_save_path, f"step_{step}.json"), "w") as f:
                json.dump(full_log, f, indent=2)
            
            #################### pre-step ####################
            # TO DO: Call LLM API to input way-points based on current drone state
            
            self.scene.step()
            # self.cam.render()
            
            
            
            # Use genesis feature to take picture with cam and save fig
            # if self.snapshots and step % self.snap_interval == 0:
            if self.snapshots:
                rgb, depth, segmentation, normal = self.cam.render(rgb=True, depth=True, segmentation=True, normal=True)
                # Convert RGB to BGR for saving
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(picture_save_path, f"snapshot_{step:04d}.png"), bgr)

                # cv2.imwrite(os.path.join(picture_save_path, f"snapshot_{step:04d}.png"), rgb)   
                # print(f"Picture saved to: {os.path.join(picture_save_path, f'snapshot_{step:04d}.png')}")
            
            def openai_api_call(self, picture_path):
                # Use OpenAI API to analyze the picture
                pass
            
            # Use OpenAI API to analyze the picture
            openai_api_call(os.path.join(picture_save_path, f"snapshot_{step:04d}.png"))
        
            step += 1

        # Save the number of steps taken (time elapsed)
    
    def rl_obs_from_state(self, drone_idx, target):
        drone = self.drones[drone_idx]
        pos = drone.get_pos()
        vel = drone.get_vel()
        quat = drone.get_quat()
        ang = drone.get_ang()

        rel_pos = torch.tensor(target, device=gs.device) - pos
        base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(quat) * self.inv_base_init_quat, quat),
            rpy=True,
            degrees=True,
        )
        inv_quat = torch.tensor([1, 0, 0, 0], device=gs.device)
        lin_vel = transform_by_quat(vel, inv_quat)
        ang_vel = transform_by_quat(ang, inv_quat)

        obs = torch.cat([
            torch.clip(rel_pos * 1/3.0, -1, 1),
            quat,
            torch.clip(lin_vel * 1/3.0, -1, 1),
            torch.clip(ang_vel * 1/3.14159, -1, 1),
            torch.zeros((4,), device=gs.device),  # dummy last actions
        ])
        return obs, base_euler

    def run(self):
        
        self.cam.start_recording()

        for i, point in enumerate(self.target_trajectory):
            self.target_markers[i].set_pos(point)
        
        # Execute the drones to fly to the current set of target points
        self.fly_to_point(state_params_save_path=self.state_params_save_path, picture_save_path=self.picture_save_path)

        
        self.cam.stop_recording(save_to_filename=self.video_save_path or "../videos/pid_flight.mp4")
        print(f"Video saved to: {self.video_save_path}")
