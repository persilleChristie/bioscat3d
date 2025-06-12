import meep as mp
import numpy as np
import matplotlib.pyplot as plt

def run_sim(rot_angle=0):
    sx = 64
    sy = 32
    resolution = 25  # pixels/μm
    cell_size = mp.Vector3(sx, sy, 0)  # for 2D
    pml_thickness = 4
    pml_layers = [
    mp.PML(thickness=pml_thickness, direction=mp.Y),
    #mp.PML(thickness=pml_thickness, direction=mp.X)
    ]
    fsrc = 1.0  # frequency of planewave (wavelength = 1/fsrc)
    n = 1  # refractive index of homogeneous material
    default_material = mp.Medium(index=n)
    k_point = mp.Vector3(y=-fsrc * n).rotate(mp.Vector3(z=1), rot_angle)

    geometry1 = [
        mp.Block(
            center=mp.Vector3(0, -sy/4),              # centered halfway down
            size=mp.Vector3(mp.inf, sy/2, mp.inf),    # fill y < 0
            material=mp.Medium(epsilon=2.56)
        )
    ]

    geometry2 = [
    mp.Cylinder(
        radius=1.0,
        height=mp.inf,                     # for 2D: extend infinitely in z
        center=mp.Vector3(0, 1),           # center at (0, 1)
        material=mp.Medium(epsilon=2.56)    # dielectric with ε=4 (or use mp.metal for PEC)
    )
    ]

    sources = [
        mp.EigenModeSource(
            src=mp.ContinuousSource(fsrc),
            center=mp.Vector3(0,sy/2 - 1.5 * pml_thickness),
            size=mp.Vector3(x=sx),
            direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
            eig_kpoint=k_point,
            eig_band=1,
            eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
            eig_match_freq=True,
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        geometry = geometry1 + geometry2,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=k_point,
        default_material=default_material,
        symmetries=[mp.Mirror(mp.Y)] if rot_angle == 0 else [],
    )

    # Define positions of flux monitors (just inside the PML boundaries)
    flux_distance_from_boundary = pml_thickness + 0.1 # µm from boundary, adjust as needed

    # Top flux region (near top PML)
    flux_top = sim.add_flux(
        fsrc, 0, 1,
        mp.FluxRegion(center=mp.Vector3(0, 0.5*sy - flux_distance_from_boundary), size=mp.Vector3(sx, 0))
    )

    flux_1 = sim.add_flux(
    fsrc, 0, 1,
    mp.FluxRegion(center=mp.Vector3(0, sy/2 - 1.5 * pml_thickness -0.1), size=mp.Vector3(sx, 0))
    )

    flux_2 = sim.add_flux(
    fsrc, 0, 1,
    mp.FluxRegion(center=mp.Vector3(0, 3), size=mp.Vector3(sx, 0))
    )

    flux_3 = sim.add_flux(
    fsrc, 0, 1,
    mp.FluxRegion(center=mp.Vector3(0, -0.1), size=mp.Vector3(sx, 0))
    )

    # Bottom flux region (near bottom PML, behind the source)
    flux_bottom = sim.add_flux(
        fsrc, 0, 1,
        mp.FluxRegion(center=mp.Vector3(0, -0.5*sy + flux_distance_from_boundary), size=mp.Vector3(sx, 0))
    )

    # Run your simulation
    sim.run(until=60)

    # Retrieve flux data
    flux_top_data = mp.get_fluxes(flux_top)
    flux_1_data = mp.get_fluxes(flux_1)
    flux_2_data = mp.get_fluxes(flux_2)
    flux_3_data = mp.get_fluxes(flux_3)
    flux_bottom_data = mp.get_fluxes(flux_bottom)

    print(f"Flux near top PML boundary: {flux_top_data[0]:.6f}")
    print(f"Flux at y = sy/2 - (pml_thickness + 2.1): {flux_1_data[0]:.6f}")
    print(f"Flux at y = 3: {flux_2_data[0]:.6f}")
    print(f"Flux at y = -0.1: {flux_3_data[0]:.6f}")
    print(f"Flux near bottom boundary: {flux_bottom_data[0]:.6f}")

    # Get field data
    ez = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)
    ex = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ex)
    ey = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ey)

    # Calculate |E| = sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2)
    abs_E = np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.imshow(abs_E.transpose(), interpolation='spline36', cmap='viridis', origin='lower')
    plt.colorbar(label='|E|')
    plt.title('Magnitude of Electric Field |E|')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()