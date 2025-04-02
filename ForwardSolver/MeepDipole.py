import meep as mp
import numpy as np

def run_dipole_on_surface(
    dipole_position=(0, 0, 0),
    dipole_direction=(0, 0, 1),
    surface_points=[],
    frequency=1.0,
    resolution=20,
    run_time=200,
    pml_thickness=1.0,
    padding=1.0
):
    # Normalize dipole direction
    d = np.array(dipole_direction)
    d = d / np.linalg.norm(d)

    # Determine primary component (Ex, Ey, Ez) based on strongest axis
    component_map = {0: mp.Ex, 1: mp.Ey, 2: mp.Ez}
    dominant_axis = np.argmax(np.abs(d))
    dipole_component = component_map[dominant_axis]

    # Determine simulation bounding box (surface points + dipole + padding)
    all_points = np.array(surface_points + [dipole_position])
    min_corner = np.min(all_points, axis=0) - pml_thickness - padding
    max_corner = np.max(all_points, axis=0) + pml_thickness + padding
    cell_size = mp.Vector3(*tuple(max_corner - min_corner))

    # Shift dipole and surface points relative to simulation origin
    origin_shift = -0.5 * (min_corner + max_corner)
    shifted_dipole = mp.Vector3(*(np.array(dipole_position) + origin_shift))
    shifted_points = [mp.Vector3(*(np.array(p) + origin_shift)) for p in surface_points]

    # Define dipole source
    source = [mp.Source(
        src=mp.ContinuousSource(frequency=frequency),
        component=dipole_component,
        center=shifted_dipole,
        amplitude=d[dominant_axis]
    )]

    # Set up simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=[mp.PML(pml_thickness)],
        sources=source,
        resolution=resolution
    )

    sim.run(until=run_time)

    # Collect E and H fields at surface points
    field_data = {}
    for comp in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
        field = [sim.get_field_point(getattr(mp, comp), pt) for pt in shifted_points]
        field_data[comp] = np.array(field)

    return field_data
