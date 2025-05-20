import genesis as gs
import math
import numpy as np
from quadcopter_controller import DronePIDController
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.vis.camera import Camera

base_rpm = 14468.429183500699
min_rpm = 0.9 * base_rpm
max_rpm = 1.5 * base_rpm


def clamp(rpm):
    return max(min_rpm, min(int(rpm), max_rpm))


def hover(drone: DroneEntity):
    drone.set_propellels_rpm([base_rpm] * 4)

def fly_to_point(target, controller: DronePIDController, scene: gs.Scene, cam):
    drone = controller.drone
    step = 0

    while step < 1000:
        drone_pos = drone.get_pos().cpu().numpy()
        distance = np.linalg.norm(np.array(target) - drone_pos)

        if distance < 0.1:
            break

        rpms = controller.update(target).cpu().numpy()
        drone.set_propellels_rpm([clamp(rpm) for rpm in rpms])

        scene.step()
        cam.render()
        step += 1
        
def main():
    gs.init(backend=gs.gpu)

    ##### Scene setup #####
    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(dt=0.01),
    )

    ##### Entities #####
    plane = scene.add_entity(morph=gs.morphs.Plane())

    drone = scene.add_entity(
        morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 0.2))
    )

    # Persistent red target marker
    target_marker = scene.add_entity(
        morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.05, fixed=True, collision=False),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0))
        ),
    )
    
    cam = scene.add_camera(
        pos=(3.5, 0.0, 2.5),
        lookat=(0.0, 0.0, 0.5),
        fov=30,
        GUI=True,
    )

    ##### Build #####
    scene.build()

    ##### PID Controller #####
    pid_params = [
        [2.0, 0.0, 0.0], 
        [2.0, 0.0, 0.0], 
        [2.0, 0.0, 0.0], 
        [20.0, 0.0, 20.0],
        [20.0, 0.0, 20.0],
        [25.0, 0.0, 20.0],
        [10.0, 0.0, 1.0],
        [10.0, 0.0, 1.0],
        [2.0, 0.0, 0.2],
    ]

    controller = DronePIDController(drone=drone, dt=0.01, base_rpm=base_rpm, pid_params=pid_params)

    ##### Trajectory #####
    cam.start_recording()
    points = [(1, 1, 2)]

    for point in points:
        target_marker.set_pos(point)  # ← update target sphere position
        fly_to_point(point, controller, scene, cam)

    cam.stop_recording(save_to_filename="../videos/fly_route.mp4")
    print("video saved ../videos/fly_route.mp4 ")
    
    # import os
    # video_path = os.path.abspath("../videos/fly_route.mp4")
    # print("Expected video path:", video_path)
    # print("File exists:", os.path.exists(video_path))

if __name__ == "__main__":
    main()
