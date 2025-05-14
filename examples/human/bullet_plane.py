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
            dt=8e-5,  # Smaller timestep for better numerical stability
            substeps=10,
            gravity=(0, 0, -9.8),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=[-1, -1, -1],
            upper_bound=[1, 1, 1],
            particle_size=0.004,
            grid_density=128,  # Higher resolution for better simulation quality
            enable_CPIC=False,  # Enable CPIC for better particle-particle coupling
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 0, 0),
            camera_lookat=(-0.1, 0.0, 0.0),
            camera_fov=35,
            max_FPS=60,
        ),
        show_viewer=True,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=False,
            show_world_frame=False,
        ),
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0, 0, -0.3),
        )
    )

    # Spherical fragment/bullet
    bullet = scene.add_entity(
        material=gs.materials.Rigid(
            rho=7850,  # Steel density
        ),
        morph=gs.morphs.Sphere(
            radius=0.0026,
            pos=(-0.8, 0.0, 0.0),
        ),
        surface=gs.surfaces.Iron(
            color=(0.7, 0.7, 0.7, 1.0),
        ),
    )

    # Human muscle tissue with our new material model
    target_muscle = scene.add_entity(
        material=gs.materials.MPM.PhaseFieldNeoHookean(
            E=0.8e6,  # Young's modulus
            nu=0.48,  # Poisson's ratio
            rho=1060.0,  # density (kg/m^3)
            l0=0.001,  # characteristic length for phase field
            residual_phase=0.02,  # residual stiffness for fully damaged material
            damage_threshold=10.0,  # threshold strain energy for damage initiation
            max_damage=1.0,  # maximum allowed damage value
            damage_rate=15.0,  # rate of damage evolution
            delete_threshold=0.0005,  # threshold for particle deletion
            one_over_sigma_c=0.1,  # inverse of critical energy release rate
        ),
        morph=gs.morphs.Box(
            size=(0.04, 0.1, 0.1),  # Made slightly thinner for easier penetration
            pos=(0.0, 0.0, 0.0),      
        ),
        surface=gs.surfaces.Rough(
            color=(0.8, 0.2, 0.2, 1.0),  # Reddish color for muscle tissue
            vis_mode="particle",
        ),
    )
    scene.build()

    # Set high velocity for the bullet to simulate fragment impact
    bullet.set_dofs_velocity((696, 0, 0, 0, 0, 0))  # 1000 m/s in x direction

    # Main simulation loop
    horizon = 3000
    # frame_counter = 0
    for i in range(horizon):
        scene.step()
        
        # 每100步打印一次详细信息
        bullet_pos = bullet.get_dofs_position()[0:3]
        bullet_vel = bullet.get_dofs_velocity()[0:3]
        print(f"Bullet position: {bullet_pos}")
        print(f"Bullet velocity: {bullet_vel}")



if __name__ == "__main__":
    main()
