import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import h5py


# Surface data and material
surface_df = pd.read_csv("SurfaceData/SurfaceData.csv")
surface_dict = surface_df.to_dict(orient='records')

x = surface_df['x'].values
y = surface_df['y'].values
z = surface_df['z'].values

plt.figure(figsize=(7, 6))
contour = plt.tricontourf(x, y, z, levels=50, cmap='plasma')
plt.colorbar(contour, label="Surface height z (μm)")
plt.title("Surface Loaded from File (Scattered Points)")
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")
plt.axis("equal")
plt.tight_layout()

# Save the figure before showing it
plt.savefig("SimData/surface_plot_pol.png", dpi=300)
print("✅ Plot saved to surface_plot_pol.png")


# Surface data and material
bump_df = pd.read_csv("SurfaceData/bumpData.csv")
bump_dict = bump_df.to_dict(orient='records')

# Define your substrate material
substrate_material = mp.Medium(epsilon=2.56)

# Material function: sum all Gaussian bumps
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

# Simulation

# Simulation parameters
width = 3
resol = 12
pml_thickness = 2
resolution = resol # pixels/um
dim = width + 2 * pml_thickness

cell_x = dim
cell_y = dim
cell_z = dim

substrate_eps = 2.56
waveLength = 325e-3
frequency = 1/waveLength

cell_size = mp.Vector3(cell_x, cell_y, cell_z)

# Plane wave incidence defined by polar and azimuthal angles (in degrees)
theta_deg = 40  # Polar angle from +Z
phi_deg = 0    # Azimuthal angle in x–y plane

# Convert to radians
theta = np.radians(theta_deg)
phi = np.radians(phi_deg)

# Build normalized k-vector
k_unit = mp.Vector3(
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    -np.cos(theta)  # negative for downward propagation
)

# Scale for Meep's eig_kpoint (|k| = frequency in Meep units)
eig_kpoint = k_unit.scale(frequency)

beta_deg = 80
beta_rad = np.radians(beta_deg)

Ey_amp = np.cos(beta_rad)
Ez_amp = np.sin(beta_rad)

sources = [
    mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=frequency, is_integrated=True),
        center=mp.Vector3(0, 0, 0.5 * cell_z - pml_thickness - 0.5),
        size=mp.Vector3(cell_x, cell_y, 0),
        direction=mp.Z,
        eig_kpoint=eig_kpoint,
        eig_band=1,
        eig_match_freq=True,
        eig_parity=mp.ODD_Y,  # Ey-dominant
        amplitude=Ey_amp
    ),
    mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=frequency, is_integrated=True),
        center=mp.Vector3(0, 0, 0.5 * cell_z - pml_thickness - 0.5),
        size=mp.Vector3(cell_x, cell_y, 0),
        direction=mp.Z,
        eig_kpoint=eig_kpoint,
        eig_band=1,
        eig_match_freq=True,
        eig_parity=mp.ODD_Z,  # Ez-dominant
        amplitude=Ez_amp
    )
]


# PMLs on all sides
pml_layers = [
    mp.PML(thickness=pml_thickness, direction=mp.X),
    mp.PML(thickness=pml_thickness, direction=mp.Y),
    mp.PML(thickness=pml_thickness, direction=mp.Z),
]

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    boundary_layers=pml_layers,
    sources=sources, # you can still add discrete objects
    default_material=bump_surface_material
)


# Add flux monitors (six directions)
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

# Run the simulation
sim.run(until=60)

# Print flux values
for name, monitor in flux_monitors.items():
    flux_val = mp.get_fluxes(monitor)[0]
    print(f"Flux through {name}: {flux_val:.6f}")

# Get all field components for total |E| calculation
ex = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, 0, cell_z), component=mp.Ex)
ey = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, 0, cell_z), component=mp.Ey)
ez = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, 0, cell_z), component=mp.Ez)

abs_E = np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)

# Plot total field magnitude
plt.figure(dpi=100)
plt.imshow(abs_E.T, cmap='inferno', origin='lower',
           extent=[-cell_x/2, cell_x/2, -cell_z/2, cell_z/2])
plt.colorbar(label='|E|')
plt.xlabel('x (µm)')
plt.ylabel('z (µm)')
plt.title('Total Electric Field Magnitude |E| (slice at y=0)')
plt.tight_layout()

# Save to file
plt.savefig("SimData/field_magnitude_slice_pol.png", dpi=300)
print("✅ Field plot saved to field_magnitude_slice_pol.png")

# Get all field components for total |E| calculation (3D volume)
ex_3d = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, cell_z), component=mp.Ex)
ey_3d = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, cell_z), component=mp.Ey)
ez_3d = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, cell_z), component=mp.Ez)

# Compute the total electric field magnitude at each (x, y, z) point
abs_E = np.sqrt(np.abs(ex_3d)**2 + np.abs(ey_3d)**2 + np.abs(ez_3d)**2)

# Parameters
sample_factor = 20  # Keep every 10th point

# Downsample the fields
ex_s = ex_3d[::sample_factor, ::sample_factor, ::sample_factor]
ey_s = ey_3d[::sample_factor, ::sample_factor, ::sample_factor]
ez_s = ez_3d[::sample_factor, ::sample_factor, ::sample_factor]
abs_E_s = abs_E[::sample_factor, ::sample_factor, ::sample_factor]

# Get spatial coordinates
nx, ny, nz = ex_s.shape
x_vals = np.linspace(-cell_x/2, cell_x/2, nx)
y_vals = np.linspace(-cell_y/2, cell_y/2, ny)
z_vals = np.linspace(-cell_z/2, cell_z/2, nz)

# Meshgrid and flatten everything
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

data = {
    "x": X.flatten(),
    "y": Y.flatten(),
    "z": Z.flatten(),
    "Ex": ex_s.flatten(),
    "Ey": ey_s.flatten(),
    "Ez": ez_s.flatten(),
    "|E|": abs_E_s.flatten()
}

df = pd.DataFrame(data)
df.to_csv("SimData/field_sampled_pol.csv", index=False)
print("✅ Saved sampled 3D field data to SimData/field_3d_sampled_pol.csv")



# Create x and z axes
x_vals = np.linspace(-cell_x/2, cell_x/2, abs_E.shape[0])
z_vals = np.linspace(-cell_z/2, cell_z/2, abs_E.shape[1])
X, Z = np.meshgrid(x_vals, z_vals)

with h5py.File("SimData/field_data_pol.h5", "w") as f:
    f.create_dataset("Ex", data=ex)
    f.create_dataset("Ey", data=ey)
    f.create_dataset("Ez", data=ez)
    f.create_dataset("abs_E", data=abs_E)
    f.create_dataset("x_vals", data=x_vals)
    f.create_dataset("z_vals", data=z_vals)
print("✅ Field data saved to SimData/field_data_pol.h5")

# Create 3D surface plot of |E|
fig = go.Figure()
fig.add_trace(go.Surface(z=abs_E.T, x=X, y=Z, colorscale='Inferno', colorbar=dict(title='|E|')))

fig.update_layout(
    title='3D Surface Plot of |E| (at y = 0)',
    scene=dict(
        xaxis_title='x (µm)',
        yaxis_title='z (µm)',
        zaxis_title='|E|',
        aspectmode='data'
    ),
    width=800,
    height=700
)

fig.write_html("SimData/field_magnitude_surface_pol.html")
print("✅ 3D surface plot saved to field_magnitude_surface_pol.html")

