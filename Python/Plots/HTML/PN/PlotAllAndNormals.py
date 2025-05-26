import pandas as pd
import numpy as np
import plotly.graph_objects as go
from glob import glob
import os

# --- Parameters ---
patch_size = 0.4
center_x = 0.0
center_y = 0.0
arrow_length = 0.01     # consistent vector length
z_aspect = 0.1          # visual compression of z-axis
name = "One"

# --- Arrow helper ---
def make_arrows(origin, vecs, color, name):
    lines_x, lines_y, lines_z = [], [], []
    for p, v in zip(origin, vecs):
        v = np.array(v)
        norm = np.linalg.norm(v)
        if norm < 1e-10 or np.isnan(norm):
            continue
        v = (v / norm) * arrow_length
        x0, y0, z0 = p
        x1, y1, z1 = p + v
        lines_x += [x0, x1, None]
        lines_y += [y0, y1, None]
        lines_z += [z0, z1, None]
    return go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode="lines",
        line=dict(color=color, width=4),
        name=name
    )

# --- Main plotting loop ---
csv_files = sorted(glob("../../../../CSV/PN/*" + name + ".csv"))
# csv_files = sorted(glob("../../../../CSV/Surface_Andreas/*.csv"))
individual_html_files = []

for file in csv_files:
    df = pd.read_csv(file)
    if not all(c in df.columns for c in ['x', 'y', 'z']):
        continue

    base = os.path.splitext(os.path.basename(file))[0]
    pos = df[['x', 'y', 'z']].values
    full_traces = [
        go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                     mode="markers", marker=dict(size=2), name="Points")
    ]

    for name, color in [('tau1', 'blue'), ('tau2', 'green'), ('normal', 'red')]:
        cols = [f"{name}_x", f"{name}_y", f"{name}_z"]
        if all(c in df.columns for c in cols):
            vecs = df[cols].values
            arrow_trace = make_arrows(pos, vecs, color, name)
            full_traces.append(arrow_trace)

    fig_full = go.Figure(data=full_traces)
    fig_full.update_layout(
        title=f"{base} — Full Field (τ + Normal)",
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=z_aspect)
        )
    )
    full_html_file = f"{base}_full_field_tau_normal.html"
    fig_full.write_html(full_html_file)
    individual_html_files.append(full_html_file)

    # --- Patch view ---
    x0 = center_x - patch_size / 2
    y0 = center_y - patch_size / 2
    patch = df[
        (df['x'] >= x0) & (df['x'] <= x0 + patch_size) &
        (df['y'] >= y0) & (df['y'] <= y0 + patch_size)
    ].copy()

    if not patch.empty:
        pos_patch = patch[['x', 'y', 'z']].values
        patch_traces = [
            go.Scatter3d(x=pos_patch[:, 0], y=pos_patch[:, 1], z=pos_patch[:, 2],
                         mode="markers", marker=dict(size=2), name="Patch Points")
        ]

        for name, color in [('tau1', 'blue'), ('tau2', 'green'), ('normal', 'red')]:
            cols = [f"{name}_x", f"{name}_y", f"{name}_z"]
            if all(c in patch.columns for c in cols):
                vecs = patch[cols].values
                arrow_trace = make_arrows(pos_patch, vecs, color, name)
                patch_traces.append(arrow_trace)

        fig_patch = go.Figure(data=patch_traces)
        fig_patch.update_layout(
            title=f"{base} — Patch (τ + Normal)",
            scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title="z",
                aspectmode="manual", aspectratio=dict(x=1, y=1, z=z_aspect)
            )
        )
        patch_html_file = f"{base}_patch_field_tau_normal.html"
        fig_patch.write_html(patch_html_file)
        individual_html_files.append(patch_html_file)

# Print result
print("Generated HTML files:")
for f in individual_html_files:
    print(" -", f)
