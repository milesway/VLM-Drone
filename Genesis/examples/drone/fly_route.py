import genesis as gs
import math
from quadcopter_controller import DronePIDController
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.vis.camera import Camera
import numpy as np

base_rpm = 14468.429183500699
min_rpm = 0.9 * base_rpm
max_rpm = 1.5 * base_rpm

def create_target_marker(scene, position):
    """Creates a red sphere mesh at the specified target location."""
    return scene.add_entity(
        morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.05),
        pos=position,
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0))
        ),
        fixed=True,
        collision=False,
    )


def hover(drone: DroneEntity):
    drone.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])


def clamp(rpm):
    return max(min_rpm, min(int(rpm), max_rpm))

    
def fly_to_point(target, controller: DronePIDController, scene: gs.Scene):
    drone = controller.drone
    step = 0

    # Add red sphere at target
    marker = create_target_marker(scene, target)

    while step < 1000:
        drone_pos = drone.get_pos().cpu().numpy()
        distance = np.linalg.norm(np.array(target) - drone_pos)

        if distance < 0.1:
            scene.remove_entity(marker)  # Remove red sphere when target is reached
            break

        rpms = controller.update(target).cpu().numpy()
        drone.set_propellels_rpm([clamp(rpm) for rpm in rpms])

        scene.step()
        step += 1


def main():
    gs.init(backend=gs.gpu)

    ##### scene #####
    scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=0.01))

    ##### entities #####
    plane = scene.add_entity(morph=gs.morphs.Plane())

    drone = scene.add_entity(morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 0.2)))

    # parameters are tuned such that the
    # drone can fly, not optimized
    pid_params = [
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

    controller = DronePIDController(drone=drone, dt=0.01, base_rpm=base_rpm, pid_params=pid_params)

    # cam = scene.add_camera(pos=(1, 1, 1), lookat=drone.morph.pos, GUI=False, res=(640, 480), fov=30)
    cam = scene.add_camera(
        pos=(3.5, 0.0, 2.5),
        lookat=(0.0, 0.0, 0.5),
        fov=30,
        GUI=True,  # or False if you want headless recording
    )
    ##### build #####

    scene.build()

    cam.start_recording()

    points = [(1, 1, 2), (-1, 2, 1), (0, 0, 0.5)]

    for point in points:
        fly_to_point(point, controller, scene)

    cam.stop_recording(save_to_filename="../videos/fly_route.mp4")


if __name__ == "__main__":
    main()
