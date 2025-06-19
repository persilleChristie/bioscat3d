import pandas as pd
import numpy as np
import plotly.graph_objects as go
from glob import glob
import os

# Parameters
patch_size = 0.4
center_x = 0.0
center_y = 0.0
arrow_length = 0.01  # fixed vector length
z_aspect = 0.1       # compress z-axis in layout only

def make_normal_arrows(origin, vecs, color, name):
    lines_x, lines_y, lines_z = [], [], []
    for p, v in zip(origin, vecs):
        # Normalize in real space
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
        mode="lines", line=dict(color=color, width=4), name=name
    )

def main():
    csv_files = sorted(glob("../../../../CSV/PN/*_PN.csv"))
    for file in csv_files:
        df = pd.read_csv(file)
        if not all(c in df.columns for c in ['x', 'y', 'z']):
            continue

        base = os.path.splitext(os.path.basename(file))[0]
        pos = df[['x', 'y', 'z']].values
        traces = [
            go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                         mode="markers", marker=dict(size=2), name="Points")
        ]

        if all(c in df.columns for c in ['normal_x', 'normal_y', 'normal_z']):
            normals = df[['normal_x', 'normal_y', 'normal_z']].values
            arrow_trace = make_normal_arrows(pos, normals, "red", "Normals")
            traces.append(arrow_trace)

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"{base} — Normals Only",
            scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title="z",
                aspectmode="manual", aspectratio=dict(x=1, y=1, z=z_aspect)
            )
        )
        fig.write_html(f"{base}_normals_only_clean.html")
        print(f"✅ Saved: {base}_normals_only_clean.html")

if __name__ == "__main__":
    main()
