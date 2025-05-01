import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import h5py
import os
import json

# --- Load surface parameters ---
with open("SurfaceData/surfaceParams.json", "r") as f:
    surfaceParams = json.load(f)

bump_dict = surfaceParams["bumpData"]
halfWidth_x = surfaceParams["halfWidth_x"]
halfWidth_y = surfaceParams["halfWidth_y"]
halfWidth_z = surfaceParams["halfWidth_z"]
resolution = surfaceParams["resolution"]
epsilon1 = surfaceParams["epsilon1"]
omega = surfaceParams["omega"]
k_vector = np.array(surfaceParams["k"])
betas = surfaceParams["betas"]
alpha = surfaceParams["alpha"]

# --- Simulation parameters ---
pml_thickness = 2

cell_x = halfWidth_x + 2 * pml_thickness
cell_y = halfWidth_y + 2 * pml_thickness
cell_z = halfWidth_z + 2 * pml_thickness

frequency = omega  # Frequency already given

cell_size = mp.Vector3(cell_x, cell_y, cell_z)

# --- Material function: surface with bumps ---
substrate_material = mp.Medium(epsilon=epsilon1)

def bump_surface_material(p):
    z_surface = 0.0
    for bump in bump_dict:
        x0 = bump["x0"]
        y0 = bump["y0"]
        h = bump["height"]
        sigma = bump["sigma"]
        r2 = (p.x - x0)**2 + (p.y - y0)**2
        z_surface += h * np.exp(-r2 / (2 * sigma**2))
    return substrate_material if p.z < z_surface else mp.air

# --- Source: Eigenmode with custom k-vector ---
sources = [
    mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=frequency, is_integrated=True),
        center=mp.Vector3(0, 0, 0.5 * cell_z - pml_thickness - 0.5),
        size=mp.Vector3(cell_x, cell_y, 0),
        direction=mp.Z,
        eig_kpoint=mp.Vector3(*k_vector),
        eig_band=1,
        eig_match_freq=True
    )
]

# --- Boundary conditions ---
pml_layers = [
    mp.PML(thickness=pml_thickness, direction=mp.X),
    mp.PML(thickness=pml_thickness, direction=mp.Y),
    mp.PML(thickness=pml_thickness, direction=mp.Z),
]

# --- Create simulation ---
sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    boundary_layers=pml_layers,
    sources=sources,
    default_material=bump_surface_material
)

# --- Add flux monitors ---
flux_regions = {
    'top':    mp.FluxRegion(center=mp.Vector3(0, 0, 0.5*cell_z - pml_thickness - 0.1), size=mp.Vector3(cell_x, cell_y, 0)),
    'bottom': mp.FluxRegion(center=mp.Vector3(0, 0, -0.5*cell_z + pml_thickness + 0.1), size=mp.Vector3(cell_x, cell_y, 0)),
    'left':   mp.FluxRegion(center=mp.Vector3(-0.5*cell_x + pml_thickness + 0.1, 0, 0), size=mp.Vector3(0, cell_y, cell_z)),
    'right':  mp.FluxRegion(center=mp.Vector3(0.5*cell_x - pml_thickness - 0.1, 0, 0), size=mp.Vector3(0, cell_y, cell_z)),
    'front':  mp.FluxRegion(center=mp.Vector3(0, -0.5*cell_y + pml_thickness + 0.1, 0), size=mp.Vector3(cell_x, 0, cell_z)),
    'back':   mp.FluxRegion(center=mp.Vector3(0, 0.5*cell_y - pml_thickness - 0.1, 0), size=mp.Vector3(cell_x, 0, cell_z)),
}

flux_monitors = {name: sim.add_flux(frequency, 0, 1, region)
                 for name, region in flux_regions.items()}

# --- Run simulation ---
sim.run(mp.at_every(10, lambda sim: print("Top flux =", mp.get_fluxes(flux_monitors["top"])[0])),
        until=200)


# --- Process flux ---
for name, monitor in flux_monitors.items():
    flux_val = mp.get_fluxes(monitor)[0]
    print(f"Flux through {name}: {flux_val:.6f}")

# --- Extract field data ---
ex = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, 0, cell_z), component=mp.Ex)
ey = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, 0, cell_z), component=mp.Ey)
ez = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, 0, cell_z), component=mp.Ez)

abs_E = np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)

# --- Save field slice ---
os.makedirs("SimData", exist_ok=True)

plt.figure(dpi=100)
plt.imshow(abs_E.T, cmap='inferno', origin='lower',
           extent=[-cell_x/2, cell_x/2, -cell_z/2, cell_z/2])
plt.colorbar(label='|E|')
plt.xlabel('x (µm)')
plt.ylabel('z (µm)')
plt.title('Total Electric Field Magnitude |E| (slice at y=0)')
plt.tight_layout()
plt.savefig("SimData/field_magnitude_slice.png", dpi=300)
plt.close()
print("✅ Field slice saved to SimData/field_magnitude_slice.png")

# --- Save 3D field ---
ex_3d = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, cell_z), component=mp.Ex)
ey_3d = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, cell_z), component=mp.Ey)
ez_3d = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, cell_z), component=mp.Ez)

abs_E_3d = np.sqrt(np.abs(ex_3d)**2 + np.abs(ey_3d)**2 + np.abs(ez_3d)**2)

with h5py.File("SimData/field_3d_data.h5", "w") as f:
    f.create_dataset("Ex", data=ex_3d)
    f.create_dataset("Ey", data=ey_3d)
    f.create_dataset("Ez", data=ez_3d)
    f.create_dataset("abs_E", data=abs_E_3d)
    f.create_dataset("x", data=np.linspace(-cell_x/2, cell_x/2, ex_3d.shape[0]))
    f.create_dataset("y", data=np.linspace(-cell_y/2, cell_y/2, ex_3d.shape[1]))
    f.create_dataset("z", data=np.linspace(-cell_z/2, cell_z/2, ex_3d.shape[2]))
print("✅ Full 3D field data saved to SimData/field_3d_data.h5")

# --- Optional: Sample and export downsampled fields ---
sample_factor = 20

ex_s = ex_3d[::sample_factor, ::sample_factor, ::sample_factor]
ey_s = ey_3d[::sample_factor, ::sample_factor, ::sample_factor]
ez_s = ez_3d[::sample_factor, ::sample_factor, ::sample_factor]
abs_E_s = abs_E_3d[::sample_factor, ::sample_factor, ::sample_factor]

nx, ny, nz = ex_s.shape
X, Y, Z = np.meshgrid(
    np.linspace(-cell_x/2, cell_x/2, nx),
    np.linspace(-cell_y/2, cell_y/2, ny),
    np.linspace(-cell_z/2, cell_z/2, nz),
    indexing="ij"
)

df = pd.DataFrame({
    "x": X.flatten(),
    "y": Y.flatten(),
    "z": Z.flatten(),
    "Ex": ex_s.flatten(),
    "Ey": ey_s.flatten(),
    "Ez": ez_s.flatten(),
    "|E|": abs_E_s.flatten()
})
df.to_csv("SimData/field_3d_sampled.csv", index=False)
print("✅ Sampled 3D field saved to SimData/field_3d_sampled.csv")

# --- Create 3D plot of |E| slice ---
fig = go.Figure()
fig.add_trace(go.Surface(
    z=abs_E.T,
    x=np.linspace(-cell_x/2, cell_x/2, abs_E.shape[0]),
    y=np.linspace(-cell_z/2, cell_z/2, abs_E.shape[1]),
    colorscale='Inferno'
))

fig.update_layout(
    title='|E| magnitude (slice at y=0)',
    scene=dict(xaxis_title="x (µm)", yaxis_title="z (µm)", zaxis_title="|E|"),
    width=800,
    height=700
)

fig.write_html("SimData/field_magnitude_surface.html")
print("✅ 3D surface plot saved to SimData/field_magnitude_surface.html")
