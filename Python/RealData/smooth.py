import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, sobel
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import os
import glob

# Parameters
sigma = 0.3             # Gaussian smoothing strength
median_size = 2         # Median filter window size

# Input/output paths
input_folder = "DataRaw"
output_folder = "DataSmooth"
os.makedirs(output_folder, exist_ok=True)

# Helper functions
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def edge_map(data):
    return np.hypot(sobel(data, axis=0), sobel(data, axis=1))

# Find .txt files
txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
if not txt_files:
    print("❌ No .txt files found in DataRaw/")
    exit(0)

for file in txt_files:
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
        header_lines = [line for line in lines if line.lstrip().startswith("#")]
        data_lines = [line for line in lines if not line.lstrip().startswith("#")]

        from io import StringIO
        data = np.loadtxt(StringIO(''.join(data_lines)))

        # Apply smoothing
        smoothed_gauss = gaussian_filter(data, sigma=sigma)
        smoothed_median = median_filter(data, size=median_size)

        # RMSE and edge preservation
        edge_original = edge_map(data)
        rmse_gauss = rmse(data, smoothed_gauss)
        rmse_median = rmse(data, smoothed_median)
        rmse_edge_gauss = rmse(edge_original, edge_map(smoothed_gauss))
        rmse_edge_median = rmse(edge_original, edge_map(smoothed_median))

        # Output filenames
        base = os.path.splitext(os.path.basename(file))[0]
        out_gauss_txt = os.path.join(output_folder, f"gaussian_{base}.txt")
        out_median_txt = os.path.join(output_folder, f"median_{base}.txt")
        out_plot = os.path.join(output_folder, f"compare_{base}.png")

        # Save smoothed .txt files with original headers
        with open(out_gauss_txt, 'w') as f:
            f.writelines(header_lines)
            np.savetxt(f, smoothed_gauss, fmt="%.6e", delimiter="\t")
        with open(out_median_txt, 'w') as f:
            f.writelines(header_lines)
            np.savetxt(f, smoothed_median, fmt="%.6e", delimiter="\t")

        # Plot comparison
        vmin, vmax = data.min(), data.max()
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title("Original")
        plt.colorbar()

        plt.subplot(2, 3, 2)
        plt.imshow(smoothed_gauss, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title("Gaussian Smoothed")
        plt.colorbar()

        plt.subplot(2, 3, 3)
        plt.imshow(smoothed_median, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title("Median Smoothed")
        plt.colorbar()

        plt.subplot(2, 3, 4)
        plt.imshow(data - smoothed_gauss, cmap='coolwarm')
        plt.title("Difference: Original - Gaussian")
        plt.colorbar()

        plt.subplot(2, 3, 5)
        plt.imshow(data - smoothed_median, cmap='coolwarm')
        plt.title("Difference: Original - Median")
        plt.colorbar()

        plt.subplot(2, 3, 6)
        plt.axis('off')
        plt.text(0.1, 0.6, f"RMSE Gaussian: {rmse_gauss:.2e}", fontsize=12)
        plt.text(0.1, 0.45, f"RMSE Median:   {rmse_median:.2e}", fontsize=12)
        plt.text(0.1, 0.3, f"Edge RMSE Gauss:  {rmse_edge_gauss:.2e}", fontsize=12)
        plt.text(0.1, 0.15, f"Edge RMSE Median: {rmse_edge_median:.2e}", fontsize=12)

        plt.tight_layout()
        plt.savefig(out_plot, dpi=300)
        plt.close()

        # Print summary to terminal
        print(f"✅ {base}.txt")
        print(f"   RMSE Gaussian:       {rmse_gauss:.3e}")
        print(f"   RMSE Median:         {rmse_median:.3e}")
        print(f"   Edge RMSE Gaussian:  {rmse_edge_gauss:.3e}")
        print(f"   Edge RMSE Median:    {rmse_edge_median:.3e}")
        print(f"   → Saved comparison to {out_plot}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
