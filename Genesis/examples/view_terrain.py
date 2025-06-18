# view_terrain.py

import genesis as gs

def main():
    gs.init()

    # Create a simple scene with visualizer enabled
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1/60, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=60,
            camera_pos=(3.0, 0.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=45,
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True
        ),
        show_viewer=True,
    )

    # Load your terrain asset
    terrain_entity = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/bullet3terrain.urdf", fixed=True)
    )

    # Optionally add a dummy object to observe terrain interaction
    sphere = scene.add_entity(gs.morphs.Sphere(radius=0.5))

    # Build and run a single simulation environment
    scene.build(n_envs=1)
    sphere.set_pos([0.0, 0.0, 2.0])  # Drop from 1m height
    
#     sphere.set_surface(
#     gs.surfaces.Rough(
#         diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.2, 0.2))
#     )
# )
    
    while True:
        scene.step()

if __name__ == "__main__":
    main()
