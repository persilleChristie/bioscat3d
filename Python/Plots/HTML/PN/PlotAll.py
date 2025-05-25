import pandas as pd
import numpy as np
import plotly.graph_objects as go
from glob import glob
import os

# Repeat the process to also generate individual plots while keeping the combined traces

# Parameters
patch_size = 0.4
center_x = 0.0
center_y = 0.0
arrow_length = 0.03
z_aspect = 2.0

def make_arrows(origin, vecs, color, name):
    lines_x, lines_y, lines_z = [], [], []
    for p, v in zip(origin, vecs):
        x0, y0, z0 = p
        x1, y1, z1 = p + v * arrow_length
        lines_x += [x0, x1, None]
        lines_y += [y0, y1, None]
        lines_z += [z0, z1, None]
    return go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode="lines",
        line=dict(color=color, width=4),
        name=name
    )

combined_full_traces = []
combined_patch_traces = []

csv_files = sorted(glob("../../../../CSV/PN/*_PN.csv"))
individual_html_files = []

for file in csv_files:
    df = pd.read_csv(file)
    if not all(c in df.columns for c in ['x', 'y', 'z']):
        continue

    base = os.path.splitext(os.path.basename(file))[0]
    pos = df[['x', 'y', 'z']].values
    full_traces = [
        go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                     mode="markers", marker=dict(size=2), name="Position")
    ]
    combined_full_traces.append(full_traces[0])

    if all(c in df.columns for c in ['tau1_x', 'tau1_y', 'tau1_z']):
        tau1 = df[['tau1_x', 'tau1_y', 'tau1_z']].values
        arrow_trace = make_arrows(pos, tau1, "blue", "tau1")
        full_traces.append(arrow_trace)
        combined_full_traces.append(arrow_trace)

    if all(c in df.columns for c in ['tau2_x', 'tau2_y', 'tau2_z']):
        tau2 = df[['tau2_x', 'tau2_y', 'tau2_z']].values
        arrow_trace = make_arrows(pos, tau2, "green", "tau2")
        full_traces.append(arrow_trace)
        combined_full_traces.append(arrow_trace)

    # Save individual full plot
    fig_full = go.Figure(data=full_traces)
    fig_full.update_layout(
        title=f"{base} — Full Field",
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=z_aspect)
        )
    )
    full_html_file = f"{base}_full_field.html"
    fig_full.write_html(full_html_file)
    individual_html_files.append(full_html_file)

    # Patch
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
                         mode="markers", marker=dict(size=2), name="Patch Position")
        ]
        combined_patch_traces.append(patch_traces[0])

        if all(c in patch.columns for c in ['tau1_x', 'tau1_y', 'tau1_z']):
            tau1 = patch[['tau1_x', 'tau1_y', 'tau1_z']].values
            arrow_trace = make_arrows(pos_patch, tau1, "blue", "tau1")
            patch_traces.append(arrow_trace)
            combined_patch_traces.append(arrow_trace)

        if all(c in patch.columns for c in ['tau2_x', 'tau2_y', 'tau2_z']):
            tau2 = patch[['tau2_x', 'tau2_y', 'tau2_z']].values
            arrow_trace = make_arrows(pos_patch, tau2, "green", "tau2")
            patch_traces.append(arrow_trace)
            combined_patch_traces.append(arrow_trace)

        # Save individual patch plot
        fig_patch = go.Figure(data=patch_traces)
        fig_patch.update_layout(
            title=f"{base} — Patch",
            scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title="z",
                aspectmode="manual", aspectratio=dict(x=1, y=1, z=z_aspect)
            )
        )
        patch_html_file = f"{base}_patch_x{center_x:.2f}_y{center_y:.2f}.html"
        fig_patch.write_html(patch_html_file)
        individual_html_files.append(patch_html_file)

# Save combined full plot
fig_combined_full = go.Figure(data=combined_full_traces)
fig_combined_full.update_layout(
    title="Combined Full Field Plot",
    scene=dict(
        xaxis_title="x", yaxis_title="y", zaxis_title="z",
        aspectmode="manual", aspectratio=dict(x=1, y=1, z=z_aspect)
    )
)
fig_combined_full.write_html("combined_full_field.html")

# Save combined patch plot
fig_combined_patch = go.Figure(data=combined_patch_traces)
fig_combined_patch.update_layout(
    title="Combined Patch Plot",
    scene=dict(
        xaxis_title="x", yaxis_title="y", zaxis_title="z",
        aspectmode="manual", aspectratio=dict(x=1, y=1, z=z_aspect)
    )
)
fig_combined_patch.write_html("combined_patch.html")

"combined_full_field.html", "combined_patch.html", individual_html_files
