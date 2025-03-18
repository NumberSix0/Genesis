
import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-7,
            substeps=10,
            gravity=(0, 0, 0),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, -0.4),
            upper_bound=(1.0, 1.0, 2.0),
            grid_density=64,
            enable_CPIC=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, 0.9, 3.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=35,
            max_FPS=120,
        ),
        show_viewer=True,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0, 0, -0.3),
        )
    )

    bullet = scene.add_entity(
        material=gs.materials.Rigid(
            rho=10000,

        ),
        morph=gs.morphs.Sphere(
            radius=0.05,
            pos=(-0.2, 0.0, 0.0),
        ),
        surface=gs.surfaces.Iron(
        ),
    )

    target_plane = scene.add_entity(
        material=gs.materials.MPM.Muscle(E=1e7, nu=0.45, rho=1000, sampler="pbs-64"),
        morph=gs.morphs.Box(
            size=(0.05, 0.5, 0.5),
            pos=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Rough(
            color=(0.6, 1.0, 0.8, 1.0),
            vis_mode="particle",
        ),
    )
    scene.build()

    bullet.set_dofs_velocity((1e3, 0, 0, 0, 0, 0))

    horizon = 3000
    for i in range(horizon):
        scene.step()
        print(bullet.get_dofs_velocity(), i, "step")


if __name__ == "__main__":
    main()
