import numpy as np
import h5py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# Load data
file_path = "SimData/field_3d_data.h5"
with h5py.File(file_path, "r") as f:
    Ex = f["Ex"][:]
    Ey = f["Ey"][:]
    Ez = f["Ez"][:]
    abs_E = f["abs_E"][:]
    x = f["x"][:]
    y = f["y"][:]
    z = f["z"][:]

# Utility: get mid-slice indices
ix_mid = len(x) // 2
iy_mid = len(y) // 2
iz_mid = len(z) // 2

output_dir = "SimData/Plots"
os.makedirs(output_dir, exist_ok=True)

# 2. Field component comparison in a slice (e.g., Ex at y=0)
components = {'Ex': Ex[:, iy_mid, :], 'Ey': Ey[:, iy_mid, :], 'Ez': Ez[:, iy_mid, :]}
for name, data in components.items():
    plt.figure(dpi=100)
    plt.imshow(np.abs(data).T, cmap="viridis", origin="lower",
               extent=[x[0], x[-1], z[0], z[-1]])
    plt.colorbar(label=f"|{name}|")
    plt.xlabel("x (µm)")
    plt.ylabel("z (µm)")
    plt.title(f"{name} magnitude slice at y=0")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_magnitude_y0.png", dpi=300)
    plt.close()

# 3. Vector field quiver (x-z plane, downsampled)
ds = 4
Xq, Zq = np.meshgrid(x[::ds], z[::ds], indexing="ij")
Ex_q = Ex[::ds, iy_mid, ::ds]
Ez_q = Ez[::ds, iy_mid, ::ds]

plt.figure(dpi=100)
plt.quiver(Xq, Zq, Ex_q, Ez_q, scale=2, color='blue')
plt.xlabel("x (µm)")
plt.ylabel("z (µm)")
plt.title("E-field vectors (x-z plane, y=0)")
plt.tight_layout()
plt.savefig(f"{output_dir}/E_quiver_xz_y0.png", dpi=300)
plt.close()

# 4. Total field magnitude slices in all 3 planes
slices = {
    "xy (z=0)": abs_E[:, :, iz_mid],
    "xz (y=0)": abs_E[:, iy_mid, :],
    "yz (x=0)": abs_E[ix_mid, :, :]
}

coords = {
    "xy (z=0)": (x, y),
    "xz (y=0)": (x, z),
    "yz (x=0)": (y, z)
}

for name, data in slices.items():
    x_ax, y_ax = coords[name]
    plt.figure(dpi=100)
    plt.imshow(data.T, cmap="inferno", origin="lower",
               extent=[x_ax[0], x_ax[-1], y_ax[0], y_ax[-1]])
    plt.colorbar(label="|E|")
    plt.xlabel(f"{name.split()[0]} (µm)")
    plt.ylabel(f"{name.split()[1]} (µm)")
    plt.title(f"Field magnitude in {name} plane")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/absE_{name.replace(' ', '_')}.png", dpi=300)
    plt.close()

# 5. Histogram of field magnitudes (global)
plt.figure(dpi=100)
plt.hist(abs_E.flatten(), bins=100, color='purple', alpha=0.7)
plt.xlabel("|E|")
plt.ylabel("Count")
plt.title("Distribution of |E| values")
plt.tight_layout()
plt.savefig(f"{output_dir}/hist_absE.png", dpi=300)
plt.close()

# 6. Interactive 3D volume rendering (Plotly, downsampled)
sample = 4
X, Y, Z = np.meshgrid(x[::sample], y[::sample], z[::sample], indexing='ij')
absE_down = abs_E[::sample, ::sample, ::sample]

fig = go.Figure(data=go.Volume(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=absE_down.flatten(),
    isomin=0.05 * abs_E.max(), isomax=abs_E.max(),
    opacity=0.1,
    surface_count=15,
    colorscale="Inferno",
))
fig.update_layout(
    title="3D Volume Rendering of |E|",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z"
    )
)
fig.write_html(f"{output_dir}/volume_absE.html")
print("✅ Advanced visualizations saved to:", output_dir)
