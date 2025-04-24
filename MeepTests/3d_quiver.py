import numpy as np
import h5py
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Load field data
file_path = "SimData/field_3d_data.h5"

with h5py.File(file_path, "r") as f:
    Ex = f["Ex"][:]
    Ey = f["Ey"][:]
    Ez = f["Ez"][:]
    x_vals = f["x"][:]
    y_vals = f["y"][:]
    z_vals = f["z"][:]

# Downsample for plotting
sample_factor = 8
Ex = Ex[::sample_factor, ::sample_factor, ::sample_factor]
Ey = Ey[::sample_factor, ::sample_factor, ::sample_factor]
Ez = Ez[::sample_factor, ::sample_factor, ::sample_factor]
x_vals = x_vals[::sample_factor]
y_vals = y_vals[::sample_factor]
z_vals = z_vals[::sample_factor]

# Create meshgrid of points
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

# Flatten the data
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = Z.flatten()
u = Ex.flatten()
v = Ey.flatten()
w = Ez.flatten()

# Normalize vector lengths for plotting
magnitude = np.sqrt(u**2 + v**2 + w**2)
nonzero = magnitude > 0
u[nonzero] /= magnitude[nonzero]
v[nonzero] /= magnitude[nonzero]
w[nonzero] /= magnitude[nonzero]

# Only plot a subset for clarity
N = 2000  # limit to 2000 arrows
indices = np.linspace(0, len(x_flat) - 1, N, dtype=int)

fig = go.Figure(data=go.Cone(
    x=x_flat[indices],
    y=y_flat[indices],
    z=z_flat[indices],
    u=u[indices],
    v=v[indices],
    w=w[indices],
    colorscale='Inferno',
    sizemode="absolute",
    sizeref=0.6,
    showscale=True,
    colorbar_title="E direction"
))

fig.update_layout(
    title="3D Quiver Plot of Electric Field Vectors (Downsampled)",
    scene=dict(
        xaxis_title='x (μm)',
        yaxis_title='y (μm)',
        zaxis_title='z (μm)',
        aspectmode='cube'
    ),
    width=900,
    height=800
)

fig.write_html("SimData/Plots/field_quiver_plot.html")
print("✅ Saved interactive 3D quiver plot to SimData/field_quiver_plot.html")
