"""
Launch a 3D oblique incidence planewave using EigenModeSource.
Reference: https://github.com/NanoComp/meep/discussions/2589
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

mp.verbosity(2)

resolution = 20
pml_thickness = 2.0
size = 10.0

cell_size = mp.Vector3(size + 2*pml_thickness,
                       size + 2*pml_thickness,
                       size + 2*pml_thickness)

pml_layers = [mp.PML(pml_thickness)]

# Incident angles in radians
theta = np.radians(40)  # from z-axis
phi = np.radians(30)    # in xy-plane

# Physical parameters
wavelength = 1.0
frequency = 1 / wavelength
n = 1.5
k_mag = n * frequency

# Wavevector components
kx = k_mag * np.sin(theta) * np.cos(phi)
ky = k_mag * np.sin(theta) * np.sin(phi)
kz = k_mag * np.cos(theta)
k_point = mp.Vector3(kx, ky, kz)
print("Wavevector k_point:", k_point)

# Set the direction perpendicular to the wavevector (2D plane)
k_dir = np.array([kx, ky, kz])
k_dir /= np.linalg.norm(k_dir)

# Choose arbitrary perpendicular vectors
if np.allclose(k_dir, [1, 0, 0]):
    perp1 = np.array([0, 1, 0])
else:
    perp1 = np.cross(k_dir, [1, 0, 0])
perp1 /= np.linalg.norm(perp1)
perp2 = np.cross(k_dir, perp1)

# Set the size along the 2 orthogonal directions only
plane_size = size
vol_size = perp1 * plane_size + perp2 * plane_size

# Zero out the dimension along k_dir (Meep requires strictly 2D volume)
zero_dim = np.argmax(np.abs(k_dir))
vol_size[zero_dim] = 0.0
vol_size = mp.Vector3(*[v if abs(v) > 1e-6 else 1/resolution for v in vol_size])

# Source definition
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

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=k_point,
    default_material=mp.Medium(index=n),
    dimensions=3,
    force_complex_fields=True
)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(), 1e-6))

# Extract and plot Ex, Ey, Ez on x=0 plane
vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, size, size))
Ex = sim.get_array(center=vol.center, size=vol.size, component=mp.Ex)
Ey = sim.get_array(center=vol.center, size=vol.size, component=mp.Ey)
Ez = sim.get_array(center=vol.center, size=vol.size, component=mp.Ez)

fig = plt.figure(figsize=(18, 5))
gs = gridspec.GridSpec(1, 3)

for idx, (field, label) in enumerate(zip([Ex, Ey, Ez], ["Ex", "Ey", "Ez"])):
    ax = fig.add_subplot(gs[idx])
    im = ax.imshow(np.real(field.T), interpolation="spline36", cmap="RdBu",
                   extent=[-0.5*size, 0.5*size, -0.5*size, 0.5*size], origin="lower")
    ax.set_title(f"Re({label}) at x = 0")
    ax.set_xlabel("y (um)")
    ax.set_ylabel("z (um)")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("planewave_3D_vector_fields.png")
plt.show()