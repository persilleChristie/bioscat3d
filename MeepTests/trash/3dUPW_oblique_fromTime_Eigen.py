"""
Demonstration of launching a planewave source at oblique incidence in 3D.
Based on: https://meep.readthedocs.io/en/latest/Python_Tutorials/Eigenmode_Source/#planewaves-in-homogeneous-media
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

mp.verbosity(2)

resolution_um = 20
pml_um = 2.0
size_um = 10.0

# Define the 3D cell
cell_size = mp.Vector3(size_um + 2 * pml_um,
                       size_um + 2 * pml_um,
                       size_um + 2 * pml_um)
pml_layers = [
    mp.PML(thickness=pml_um, direction=mp.X),
    mp.PML(thickness=pml_um, direction=mp.Y),
    mp.PML(thickness=pml_um, direction=mp.Z)
]

# Incident angle in degrees, converted to radians
incident_theta = np.radians(40.0)  # polar angle from +x axis
incident_phi = np.radians(30.0)    # azimuthal angle from x-axis in xy-plane

# Source parameters
wavelength_um = 1.0
frequency = 1 / wavelength_um
n_mat = 1.5  # refractive index

# Define the wavevector direction (k_point) in 3D
k_mag = n_mat * frequency
k_x = k_mag * np.sin(incident_theta) * np.cos(incident_phi)
k_y = k_mag * np.sin(incident_theta) * np.sin(incident_phi)
k_z = k_mag * np.cos(incident_theta)
k_point = mp.Vector3(k_x, k_y, k_z)

# Print the wavevector direction
print("Wavevector k_point:", k_point)

# Define eigenmode source direction based on k_point (normalize to find propagation axis)
k_dir = np.array([k_x, k_y, k_z])
k_dir /= np.linalg.norm(k_dir)

# Determine eig_vol perpendicular to wavevector direction
# Choose a large area perpendicular to k_dir
perp1 = np.cross(k_dir, [1, 0, 0])
if np.linalg.norm(perp1) < 1e-6:
    perp1 = np.cross(k_dir, [0, 1, 0])
perp1 /= np.linalg.norm(perp1)
perp2 = np.cross(k_dir, perp1)
perp2 /= np.linalg.norm(perp2)

# Project vol_size onto plane perpendicular to k_dir (force zero size along propagation direction)
vol_vec = perp1 * size_um + perp2 * size_um
# Normalize so we get a plane shape in sim coordinates
vol_size = mp.Vector3(
    vol_vec[0] if abs(k_dir[0]) < 0.99 else 0.0,
    vol_vec[1] if abs(k_dir[1]) < 0.99 else 0.0,
    vol_vec[2] if abs(k_dir[2]) < 0.99 else 0.0
)

# Force vol_size to be strictly 2D (set smallest component to 0)
sizes = np.array([vol_size.x, vol_size.y, vol_size.z])
min_idx = np.argmin(np.abs(sizes))
sizes[min_idx] = 0.0
# Ensure remaining two directions are at least 1/resolution_um
sizes[sizes < 1e-6] = 1 / resolution_um
vol_size = mp.Vector3(*sizes)



# Define the eigenmode source
sources = [
    mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=frequency),
        center=mp.Vector3(),
        size=vol_size,
        direction=mp.NO_DIRECTION,
        eig_kpoint=k_point,
        eig_band=1,
        eig_parity=mp.ODD_Z,
        eig_vol=mp.Volume(center=mp.Vector3(), size=vol_size),
    )
]

# Set up the simulation
sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution_um,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=k_point,
    default_material=mp.Medium(index=n_mat),
    symmetries=[],
    dimensions=3,
    force_complex_fields=True
)

# Run the simulation until field decay
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(), 1e-6))

# Output: get 2D slices of Ex, Ey, Ez at center x plane
vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, size_um, size_um))
Ex = sim.get_array(center=vol.center, size=vol.size, component=mp.Ex)
Ey = sim.get_array(center=vol.center, size=vol.size, component=mp.Ey)
Ez = sim.get_array(center=vol.center, size=vol.size, component=mp.Ez)

# Plot Ex, Ey, Ez in y-z plane (at x=0)
fig = plt.figure(figsize=(18, 5))
gs = gridspec.GridSpec(1, 3)

for idx, (field, label) in enumerate(zip([Ex, Ey, Ez], ["Ex", "Ey", "Ez"])):
    ax = fig.add_subplot(gs[idx])
    im = ax.imshow(np.real(field.T), interpolation="spline36", cmap="RdBu",
                   extent=[-0.5*size_um, 0.5*size_um, -0.5*size_um, 0.5*size_um], origin="lower")
    ax.set_title(f"Re({label}) at x = 0")
    ax.set_xlabel("y (um)")
    ax.set_ylabel("z (um)")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("planewave_3D_vector_fields.png")
plt.show()
