import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import plotly.express as px
import plotly.io as pio

# --- Helper Functions ---
def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2, axis=1)

def angular_diff(v1, v2):
    dot = np.sum(v1 * v2, axis=1)
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cos_theta = np.clip(dot / norms, -1.0, 1.0)
    return np.arccos(cos_theta) * (180 / np.pi)

# --- Match files by same basename ---
andreas_dir = "../../../../CSV/Andreas/"
pn_dir = "../../../../CSV/PN/"

andreas_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob(os.path.join(andreas_dir, "*.csv"))}
pn_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob(os.path.join(pn_dir, "*.csv"))}

common_keys = set(andreas_files.keys()) & set(pn_files.keys())

# --- Process matching pairs ---
results = []
for name in sorted(common_keys):
    f1 = andreas_files[name]
    f2 = pn_files[name]
    print(f"🔍 Comparing {f1} <-> {f2}")
    
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)

    if df1.shape != df2.shape:
        print(f"⚠️ Shape mismatch in {name}, skipping.")
        continue

    # Compare positions
    pos1 = df1[['x', 'y', 'z']].values
    pos2 = df2[['x', 'y', 'z']].values
    d_pos = euclidean_dist(pos1, pos2)

    # Compare vectors
    tau1_a = df1[['tau1_x', 'tau1_y', 'tau1_z']].values
    tau1_b = df2[['tau1_x', 'tau1_y', 'tau1_z']].values
    d_tau1 = angular_diff(tau1_a, tau1_b)

    tau2_a = df1[['tau2_x', 'tau2_y', 'tau2_z']].values
    tau2_b = df2[['tau2_x', 'tau2_y', 'tau2_z']].values
    d_tau2 = angular_diff(tau2_a, tau2_b)

    norm1 = df1[['normal_x', 'normal_y', 'normal_z']].values
    norm2 = df2[['normal_x', 'normal_y', 'normal_z']].values
    d_norm = angular_diff(norm1, norm2)

    results.append({
        'pair': name,
        'RMSE position': np.sqrt(np.mean(d_pos**2)),
        'mean angle tau1 (deg)': np.mean(d_tau1),
        'mean angle tau2 (deg)': np.mean(d_tau2),
        'mean angle normal (deg)': np.mean(d_norm),
    })

    # Plot histogram
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].hist(d_pos, bins=50, color='gray')
    axs[0].set_title('Position Error')
    axs[1].hist(d_tau1, bins=50, color='blue')
    axs[1].set_title('Tau1 Angle Diff (°)')
    axs[2].hist(d_tau2, bins=50, color='green')
    axs[2].set_title('Tau2 Angle Diff (°)')
    axs[3].hist(d_norm, bins=50, color='red')
    axs[3].set_title('Normal Angle Diff (°)')
    fig.suptitle(f'Differences for: {name}', fontsize=14)
    fig.tight_layout()
    fig.savefig(f"comparison_{name}.png")
    plt.close(fig)

# --- Save summary CSV ---
results_df = pd.DataFrame(results)
results_df.to_csv("comparison_summary.csv", index=False)
print("✅ Done! Saved to comparison_summary.csv and comparison_*.png")

# --- Plot interactive summary ---
df = pd.read_csv("comparison_summary.csv")

fig = px.bar(
    df.melt(id_vars="pair", var_name="Metric", value_name="Value"),
    x="pair", y="Value", color="Metric",
    barmode="group",
    title="Comparison Summary Metrics by Pair"
)

output_file = "comparison_summary_plot.html"
pio.write_html(fig, file=output_file, auto_open=True)

print(f"✅ Saved interactive plot to {output_file}")
