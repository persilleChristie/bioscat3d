import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

name = "Zero"

# Find all CSV files starting with 'tangential_error' in the ../../CSV directory
csv_files = sorted(glob.glob("../../CSV/tangential_error*_" + name + ".csv"))

# Separate files for tangent 1 and tangent 2
tangent1_files = [f for f in csv_files if "tangential_error1" in f]
tangent2_files = [f for f in csv_files if "tangential_error2" in f]

# Read each file separately for tangent 1 and tangent 2
dfs1 = [pd.read_csv(f, header=None) for f in tangent1_files]
dfs2 = [pd.read_csv(f, header=None) for f in tangent2_files]

# Plot each tangent 1 file separately
for i, df in enumerate(dfs1):
    error = df[0].values
    plt.figure()
    plt.plot(error)
    plt.xlabel("Sample Index")
    plt.ylabel("Error")
    plt.title(f"Tangent 1 Error (Surface {name})")
    plt.tight_layout()
    plt.savefig(f"tangent1_error_plot_{name}.png")
    plt.show()

# Plot each tangent 2 file separately
for i, df in enumerate(dfs2):
    error = df[0].values
    plt.figure()
    plt.plot(error)
    plt.xlabel("Sample Index")
    plt.ylabel("Error")
    plt.title(f"Tangent 2 Error (Surface {name})")
    plt.tight_layout()
    plt.savefig(f"tangent2_error_plot_{name}.png")
    plt.show()

# Histogram of the errors
error1 = np.concatenate([df[0].values for df in dfs1]) if dfs1 else np.array([])
error2 = np.concatenate([df[0].values for df in dfs2]) if dfs2 else np.array([])

# Plot histogram for each pair of tangent 1 and tangent 2 files
num_pairs = min(len(dfs1), len(dfs2))
for i in range(num_pairs):
    error1 = dfs1[i][0].values
    error2 = dfs2[i][0].values
    plt.figure()
    plt.hist(error1, bins=50, alpha=0.5, label="Tangent 1")
    plt.hist(error2, bins=50, alpha=0.5, label="Tangent 2")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Tangential Errors (Surface {name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"tangential_error_histogram_pair_{name}.png")
    plt.show()



