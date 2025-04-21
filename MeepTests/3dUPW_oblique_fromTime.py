import meep as mp
import numpy as np
import cmath
import matplotlib.pyplot as plt

# --- Parameters ---
lambda0 = 0.325                # Wavelength in μm
fcen = 1 / lambda0             # Frequency in 1/μm
theta_deg = 30                 # Polar angle in degrees
phi_deg = 45                   # Azimuthal angle in degrees
resolution = 10                # Pixels/μm
dpml = 1.0                     # PML thickness
sx, sy, sz = 5, 5, 5           # Domain size in μm

cell = mp.Vector3(sx, sy, sz)
pml_layers = [mp.PML(dpml)]

# --- Wavevector components ---
theta = np.radians(theta_deg)
phi = np.radians(phi_deg)
k_mag = 2 * np.pi * fcen
k_vec = k_mag * np.array([
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta)
])
kx, ky, kz = k_vec

# --- Plane wave amp_func ---
def plane_wave(pos):
    return cmath.exp(1j * (kx * pos.x + ky * pos.y + kz * pos.z))

# --- Source plane perpendicular to k vector (inject from -x face) ---
sources = [
    mp.Source(
        src=mp.ContinuousSource(frequency=fcen),
        component=mp.Ey,
        center=mp.Vector3(-0.5 * sx + dpml + 0.1),
        size=mp.Vector3(0, sy, sz),
        amp_func=plane_wave
    )
]

# --- Simulation ---
sim = mp.Simulation(
    cell_size=cell,
    resolution=resolution,
    boundary_layers=pml_layers,
    sources=sources,
    geometry=[],
    dimensions=3,
    force_complex_fields=True
)

# --- Add DFT monitor ---
dft_obj = sim.add_dft_fields(
    [mp.Ex, mp.Ey, mp.Ez],
    fcen, 0, 1,
    center=mp.Vector3(),
    size=mp.Vector3(sx, sy, sz)
)

# --- Run ---
sim.run(until_after_sources=mp.stop_when_fields_decayed(
    50, mp.Ey, mp.Vector3(), 1e-4
))

# --- Get frequency-domain fields ---
Ey = sim.get_dft_array(dft_obj, mp.Ey, 0)

# --- Slice and plot phase of Ey in x-z plane at y = 0 ---
slice_y = Ey[:, Ey.shape[1] // 2, :]
phase = np.angle(slice_y)

x = np.linspace(-sx / 2, sx / 2, slice_y.shape[0])
z = np.linspace(-sz / 2, sz / 2, slice_y.shape[1])
X, Z = np.meshgrid(x, z, indexing='ij')

plt.figure(figsize=(6, 5))
plt.imshow(phase.T, extent=[-sx/2, sx/2, -sz/2, sz/2], cmap='twilight', origin='lower')
plt.colorbar(label='Phase [radians]')
plt.title(f"Phase arg(Ey) in x–z plane (y = 0)\nθ={theta_deg}°, φ={phi_deg}°")
plt.xlabel("x (μm)")
plt.ylabel("z (μm)")
plt.tight_layout()
plt.show()
