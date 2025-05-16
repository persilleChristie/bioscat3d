import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("./SimData/flux_comparison.csv")

# Ensure output directory exists
os.makedirs("SimData/plots_by_monitor", exist_ok=True)

# Unique monitor names
monitors = df["monitor"].unique()

# Plot for each monitor
for monitor in monitors:
    monitor_df = df[df["monitor"] == monitor].copy()

    # Normalize by the maximum absolute scattered flux
    max_flux = monitor_df["scattered_flux"].abs().max()
    monitor_df["norm_scattered"] = monitor_df["scattered_flux"] / max_flux

    # Plot
    plt.figure(dpi=120)
    plt.plot(np.degrees(monitor_df["beta (rad)"]),
             monitor_df["norm_scattered"],
             marker="o",
             label=f"{monitor}")

    plt.xlabel("Polarization angle β (degrees)")
    plt.ylabel("Scattered flux (normalized)")
    plt.title(f"Normalized scattered flux vs β for monitor: {monitor}")
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"SimData/plots_by_monitor/{monitor}_norm_scattered_vs_beta.png")
    plt.close()

print("✅ Saved plots to SimData/plots_by_monitor/")
