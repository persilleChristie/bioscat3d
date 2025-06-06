import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import re

name = "Zero"
name += "_014"

# Find all CSV files starting with 'tangential_error' in the ../../CSV directory
csv_files = sorted(glob.glob("../../CSV/*_tangential_error*_" + name + ".csv"))

# Separate files for tangent 1 and tangent 2
tangent1_files = [f for f in csv_files if "tangential_error1" in f]
tangent2_files = [f for f in csv_files if "tangential_error2" in f]

# Read each file separately for tangent 1 and tangent 2
# dfs1 = [pd.read_csv(f, header=None) for f in tangent1_files]
# dfs2 = [pd.read_csv(f, header=None) for f in tangent2_files]


def get_field_type(filename):
    """Extracts E or H from filename if present."""
    basename = os.path.basename(filename)
    match = re.search(r'([EH])_tangential_error[12]', basename)
    return match.group(1) if match else "Unknown"

# Plot each tangent 1 file separately
for i, file in enumerate(tangent1_files):
    df = pd.read_csv(file, header=None)
    error = df[0].values
    field_type = get_field_type(file)


    plt.figure()
    plt.plot(error)
    plt.xlabel("Sample Index")
    plt.ylabel("Error")
    plt.title(f"Tangent 1 Error, {field_type} field (Surface {name})")
    plt.tight_layout()
    plt.savefig(f"tangent1_error_plot_{name}.png")
    plt.show()

# Plot each tangent 2 file separately
for i, file in enumerate(tangent2_files):
    df = pd.read_csv(file, header=None)
    error = df[0].values
    field_type = get_field_type(file)


    plt.figure()
    plt.plot(error)
    plt.xlabel("Sample Index")
    plt.ylabel("Error")
    plt.title(f"Tangent 2 Error, {field_type} field (Surface {name})")
    plt.tight_layout()
    plt.savefig(f"tangent2_error_plot_{name}.png")
    plt.show()

# # Histogram of the errors
# error1 = np.concatenate([df[0].values for df in dfs1]) if dfs1 else np.array([])
# error2 = np.concatenate([df[0].values for df in dfs2]) if dfs2 else np.array([])

# # Plot histogram for each pair of tangent 1 and tangent 2 files
# num_pairs = min(len(dfs1), len(dfs2))
# for i in range(num_pairs):
#     error1 = dfs1[i][0].values
#     error2 = dfs2[i][0].values
#     plt.figure()
#     plt.hist(error1, bins=50, alpha=0.5, label="Tangent 1")
#     plt.hist(error2, bins=50, alpha=0.5, label="Tangent 2")
#     plt.xlabel("Error")
#     plt.ylabel("Frequency")
#     plt.title(f"Histogram of Tangential Errors (Surface {name})")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"tangential_error_histogram_pair_{name}.png")
#     plt.show()

paired_files = []
for file1 in tangent1_files:
    field_type1 = get_field_type(file1)
    for file2 in tangent2_files:
        field_type2 = get_field_type(file2)
        if field_type1 == field_type2:
            paired_files.append((file1, file2, field_type1))
            tangent2_files.remove(file2)  # Prevent reuse
            break

# Plot histograms for each pair
for i, (file1, file2, field_type) in enumerate(paired_files):
    error1 = pd.read_csv(file1, header=None)[0].values
    error2 = pd.read_csv(file2, header=None)[0].values
    
    plt.figure()
    plt.hist(error1, bins=50, alpha=0.5, label="Tangent 1")
    plt.hist(error2, bins=50, alpha=0.5, label="Tangent 2")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Tangential Errors ({field_type}-field, Surface {name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"tangential_error_histogram_{field_type}_{name}_{i}.png")
    plt.show()


