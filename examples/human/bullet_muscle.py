import numpy as np

import genesis as gs


def main():
    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=8e-5,  # Smaller timestep for better numerical stability
            substeps=10,
            gravity=(0, 0, 0),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=[-1.0, -1.0, -1.0],
            upper_bound=[1.0, 1.2, 1.0],
            # particle_size=0.002,
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
            show_world_frame=False
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
            pos=(0.02, 1.0, 0.1),
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
        morph=gs.morphs.Mesh(
            file="model/muscle.obj",
        ),
        surface=gs.surfaces.Rough(
            color=(0.8, 0.2, 0.2, 1.0),  # Reddish color for muscle tissue
            vis_mode="particle",
        ),
    )
    scene.build()

    # Set high velocity for the bullet to simulate fragment impact
    bullet.set_dofs_velocity((0, 0, -696, 0, 0, 0))  # 1000 m/s in x direction

    # Main simulation loop
    horizon = 600

    init_distance = [0]
    step_distance = [0]
    ad = True

    for i in range(horizon):
        scene.step()
        
        bullet_pos = bullet.get_dofs_position()[0:3]
        bullet_vel = bullet.get_dofs_velocity()[0:3]
        print(f"Bullet velocity: {bullet_vel}")
        if bullet_vel[2].item() > -696:
            if ad:
                init_distance[0] = bullet_pos[2].item()
                ad = False
            else:
                step_distance[0] = bullet_pos[2].item()
                print(f"Bullet position: {init_distance[0] - step_distance[0]}")
        # Get damage data for visualization and analysis
        damage_field = target_muscle.get_state().damage
        active_field = target_muscle.get_state().active
        #打印活动/非活动粒子的数量以及它们的最大damage值
        inactive_mask = active_field == 0
        print(f"Max damage: {gs.torch.min(damage_field)}")
        if gs.torch.sum(inactive_mask) > 0:
            inactive_damage = damage_field[inactive_mask]
            print(f"Inactive particles: {gs.torch.sum(inactive_mask)}, Max damage: {gs.torch.max(inactive_damage) if len(inactive_damage) > 0 else 0}")
        

if __name__ == "__main__":
    main()
