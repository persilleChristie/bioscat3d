import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.pyplot as plt
import os
import glob
from io import StringIO

# Parameters
sigma = 0.3
median_size = 3
bins = 200

# Paths
input_folder = "DataRaw"
output_folder = "DataSmooth"
os.makedirs(output_folder, exist_ok=True)

# Utility
def load_with_header(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    header = [line for line in lines if line.lstrip().startswith("#")]
    data = [line for line in lines if not line.lstrip().startswith("#")]
    return header, np.loadtxt(StringIO("".join(data)))

# Files
txt_files = glob.glob(os.path.join(input_folder, "*.txt"))

if not txt_files:
    print("❌ No .txt files found in DataRaw/")
    exit(0)

for file in txt_files:
    try:
        base = os.path.splitext(os.path.basename(file))[0]
        header, data = load_with_header(file)

        # Smooth
        smoothed_gauss = gaussian_filter(data, sigma=sigma)
        smoothed_median = median_filter(data, size=median_size)

        # Histogram range based on full range across all
        combined = np.concatenate([data.flatten(), smoothed_gauss.flatten(), smoothed_median.flatten()])
        hist_range = (combined.min(), combined.max())

        # Histogram bins (shared)
        hist_data, bin_edges = np.histogram(data, bins=bins, range=hist_range, density=True)
        hist_gauss, _ = np.histogram(smoothed_gauss, bins=bins, range=hist_range, density=True)
        hist_median, _ = np.histogram(smoothed_median, bins=bins, range=hist_range, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot 1: Original vs Gaussian
        plt.figure(figsize=(7, 5))
        plt.plot(bin_centers, hist_data, label="Original", color='black', linewidth=1.5)
        plt.plot(bin_centers, hist_gauss, label="Gaussian", color='green', linestyle='--')
        plt.title(f"{base}: Original vs Gaussian")
        plt.xlabel("Height Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"hist_pair_gauss_{base}.png"), dpi=300)
        plt.close()

        # Plot 2: Original vs Median
        plt.figure(figsize=(7, 5))
        plt.plot(bin_centers, hist_data, label="Original", color='black', linewidth=1.5)
        plt.plot(bin_centers, hist_median, label="Median", color='orange', linestyle='--')
        plt.title(f"{base}: Original vs Median")
        plt.xlabel("Height Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"hist_pair_median_{base}.png"), dpi=300)
        plt.close()

        print(f"✅ Histograms saved for {base}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
