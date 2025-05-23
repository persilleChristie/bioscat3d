import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# Find all CSV files starting with 'tangential_error' in the ../../CSV directory
csv_files = sorted(glob.glob("../../CSV/tangential_error*.csv"))

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
    plt.title(f"Tangent 1 Error (File {i+1})")
    plt.tight_layout()
    plt.savefig(f"tangent1_error_plot_{i+1}.png")
    plt.show()

# Plot each tangent 2 file separately
for i, df in enumerate(dfs2):
    error = df[0].values
    plt.figure()
    plt.plot(error)
    plt.xlabel("Sample Index")
    plt.ylabel("Error")
    plt.title(f"Tangent 2 Error (File {i+1})")
    plt.tight_layout()
    plt.savefig(f"tangent2_error_plot_{i+1}.png")
    plt.show()

# Concatenate all tangent 1 and tangent 2 DataFrames
df1 = pd.concat(dfs1, ignore_index=True)
df2 = pd.concat(dfs2, ignore_index=True)

# Extract error arrays for plotting
error1 = df1[0].values
error2 = df2[0].values

# Plot all tangential errors together
plt.figure()
plt.plot(error1, label="Tangent 1")
plt.plot(error2, label="Tangent 2")
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.title("Tangential Errors")
plt.legend()
plt.tight_layout()
plt.savefig("tangential_error_plot.png")
plt.show()

# Plot and save individually for each tangent
plt.figure()
plt.plot(error1)
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.title("Tangent 1 Error")
plt.tight_layout()
plt.savefig("tangent1_error_plot.png")
plt.show()

plt.figure()
plt.plot(error2)
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.title("Tangent 2 Error")
plt.tight_layout()
plt.savefig("tangent2_error_plot.png")
plt.show()

# Histogram of the errors
plt.figure()
plt.hist(error1, bins=50, alpha=0.5, label="Tangent 1")
plt.hist(error2, bins=50, alpha=0.5, label="Tangent 2")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Histogram of Tangential Errors")
plt.legend()
plt.tight_layout()
plt.savefig("tangential_error_histogram.png")
plt.show()



