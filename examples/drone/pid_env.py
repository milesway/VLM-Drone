import genesis as gs
import numpy as np
from genesis.engine.entities.drone_entity import DroneEntity
from quadcopter_controller import DronePIDController


class PidEnv:
    def __init__(self, target_trajectory, pid_params=None, show_viewer=True, record=False):
        self.base_rpm = 14468.429183500699
        self.min_rpm = 0.9 * self.base_rpm
        self.max_rpm = 1.5 * self.base_rpm
        self.target_trajectory = target_trajectory
        self.record = record
        self.pid_params = pid_params or self.default_pid_params()

        gs.init(backend=gs.gpu)

        self.scene = gs.Scene(
            show_viewer=show_viewer,
            sim_options=gs.options.SimOptions(dt=0.01),
        )

        self._setup_scene()
        self._setup_drone()
        self._setup_camera()

        self.scene.build()

    def default_pid_params(self):
        return [
            [2.0, 0.0, 0.0],  # pos_x
            [2.0, 0.0, 0.0],  # pos_y
            [2.0, 0.0, 0.0],  # pos_z
            [20.0, 0.0, 20.0],
            [20.0, 0.0, 20.0],
            [25.0, 0.0, 20.0],
            [10.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],
            [2.0, 0.0, 0.2],
        ]
        
    def set_pid_params(self, new_params):
        self.controller.set_pid_params(new_params)
        
    def _setup_scene(self):
        self.scene.add_entity(morph=gs.morphs.Plane())

        self.target_marker = self.scene.add_entity(
            morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.05, fixed=True, collision=False),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0))
            ),
        )

    def _setup_drone(self):
        self.drone = self.scene.add_entity(
            morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 0.2))
        )
        self.controller = DronePIDController(
            drone=self.drone, dt=0.01, base_rpm=self.base_rpm, pid_params=self.pid_params
        )

    def _setup_camera(self):
        self.cam = self.scene.add_camera(
            pos=(3.5, 0.0, 2.5),
            lookat=(0.0, 0.0, 0.5),
            fov=30,
            GUI=True,
        )

    def clamp(self, rpm):
        return max(self.min_rpm, min(int(rpm), self.max_rpm))

    def fly_to_point(self, target):
        step = 0
        while step < 1000:
            pos = self.drone.get_pos().cpu().numpy()
            if np.linalg.norm(np.array(target) - pos) < 0.1:
                break

            rpms = self.controller.update(target).cpu().numpy()
            self.drone.set_propellels_rpm([self.clamp(rpm) for rpm in rpms])

            self.scene.step()
            self.cam.render()
            step += 1

    def run(self, save_video_path=None):
        if self.record:
            self.cam.start_recording()

        for point in self.target_trajectory:
            self.target_marker.set_pos(point)
            self.fly_to_point(point)

        if self.record:
            save_path = save_video_path or "../videos/pid_flight.mp4"
            self.cam.stop_recording(save_to_filename=save_path)
            print(f"Video saved to: {save_path}")
