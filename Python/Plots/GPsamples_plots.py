import os
import re
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import defaultdict

csv_folder = "../../CSV/GPsamples/"
pattern = re.compile(r"(SE|Matern)_sample\d+_l_([0-9.]+)_tau_([0-9.]+)(?:_p_([0-9.]+))?\.csv")

# Group files by (kernel, tau, p, l)
files_by_params = defaultdict(list)
for filepath in glob.glob(os.path.join(csv_folder, "*.csv")):
    filename = os.path.basename(filepath)
    match = pattern.match(filename)
    if match:
        kernel, l_val, tau_val, p_val = match.groups()
        l_val = float(l_val)
        tau_val = float(tau_val)
        p_key = float(p_val) if p_val is not None else None
        files_by_params[(kernel, tau_val, p_key, l_val)].append(filepath)

# Group by (kernel, tau, p), then per l
grouped = defaultdict(list)
for (kernel, tau, p, l), files in files_by_params.items():
    grouped[(kernel, tau, p)].append((l, files))

# Plot each group
for (kernel, tau, p), l_groups in grouped.items():
    l_groups.sort(key=lambda x: x[0])  # sort by l

    num_l = len(l_groups)
    max_samples = max(len(files) for _, files in l_groups)

    fig = plt.figure(figsize=(max_samples * 3, num_l * 3))
    outer_grid = gridspec.GridSpec(num_l, 1, hspace=0.5)

    last_im = None  # for colorbar

    for row_idx, (l_val, sample_files) in enumerate(l_groups):
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            1, len(sample_files),
            subplot_spec=outer_grid[row_idx],
            wspace=0.2
        )

        for col_idx, file in enumerate(sample_files):
            data = pd.read_csv(file, header=None).values
            ax = plt.Subplot(fig, inner_grid[col_idx])
            im = ax.imshow(data, aspect='auto', cmap='viridis', extent=[0, 5, 0, 5], origin='lower')
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 5)
            ax.tick_params(labelsize=12)
            fig.add_subplot(ax)
            last_im = im

        # Add a centered title above the row
        row_ax = fig.add_subplot(outer_grid[row_idx])
        row_ax.axis('off')
        row_ax.set_title(r'$\ell$ = ' + f"{l_val:.2f}", fontsize=16, pad=20)

    # Shared colorbar on right
    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=12)

    # Main title
    title = f"{kernel} samples (tau={tau/100}"
    if p is not None:
        title += f", p={p}"
    title += ")"
    
    fig.subplots_adjust(left=0.05, right=0.9, top=0.88, bottom=0.05)
    fig.suptitle(title, fontsize=20, y=0.95)


    # Save
    fname = f"ResultsGPSamples/{kernel}_heatmaps_tau_{int(tau)}" + (f"_p_{p}" if p is not None else "") + ".png"
    plt.savefig(fname)
    plt.close()
