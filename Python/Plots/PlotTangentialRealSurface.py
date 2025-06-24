import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
    "figure.titlesize": 20
})

# Set your root directory
root_dir = "../../CSV/TangentialErrors_RealSurface1"  # Change surface number as needed
output_dir = "./Surface1_heatmaps"  # Folder where plots will be saved
os.makedirs(output_dir, exist_ok=True)

# Updated regex to match new filename format
filename_pattern = re.compile(r"([EH]_tangential_error[12])__lambda(\d{3})_beta(\d{3})\.csv")

# Store data: error_type -> lambda -> beta -> max_error
grouped_errors = defaultdict(lambda: defaultdict(dict))

# Loop over files in directory
for filename in os.listdir(root_dir):
    match = filename_pattern.search(filename)
    if match:
        error_type = match.group(1)
        lambda_val = int(match.group(2)) / 1000.0  # Convert back to float
        beta_val = int(match.group(3)) / 1000.0

        # Load the CSV file (one column)
        data = pd.read_csv(os.path.join(root_dir, filename), header=None).squeeze()
        max_error = data.max()

        grouped_errors[error_type][lambda_val][beta_val] = max_error
    else:
        print(f"Skipped (no match): {filename}")


# Gather all unique lambdas and betas
all_lambdas = sorted({l for e in grouped_errors.values() for l in e})
all_betas = sorted({b for e in grouped_errors.values() for v in e.values() for b in v})

# Define colormap and normalization
colors = ["steelblue", "whitesmoke", "tomato"]
cmap = LinearSegmentedColormap.from_list("threshold_cmap", colors)

# Generate and save heatmaps
for error_type, error_data in grouped_errors.items():
    error_matrix = np.array([
        [error_data.get(lmb, {}).get(beta, np.nan) for beta in all_betas]
        for lmb in all_lambdas
    ])

    tau_name = r'$\tau_1$' if error_type[-1] == "1" else r'$\tau_2$'

    plt.figure(figsize=(10, 6))
    sns.heatmap(error_matrix, annot=True, fmt=".2e",
                xticklabels=all_betas, yticklabels=all_lambdas,
                cmap=cmap)
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\lambda$')
    plt.title(f'Heatmap of $e_{{\\max}}$ ({tau_name}, Auxiliary points per $\lambda$: ${5}$)')
    plt.tight_layout()

    # Save to file
    filename_out = f"{error_type}_heatmap.png"
    plt.savefig(os.path.join(output_dir, filename_out), dpi=300)
    plt.close()  # Close figure to save memory

