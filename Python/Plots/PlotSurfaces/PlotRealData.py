import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

surfacePath = "../../RealData/DataRaw/"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
    "figure.titlesize": 22
})


def load_topography_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        data_lines = [line for line in lines if not line.startswith('#')]
        data = np.array([[float(val) for val in line.split()] for line in data_lines])
    
    # Extract dimensions from header
    width = height = None
    for line in lines:
        if line.startswith("# Width:"):
            width = float(line.split()[2])
        if line.startswith("# Height:"):
            height = float(line.split()[2])
    
    if width is None or height is None:
        raise ValueError("Width/Height metadata not found.")
    
    return data, width, height

# Load data
data1, width1, height1 = load_topography_data(surfacePath + "1.txt")
data2, width2, height2 = load_topography_data(surfacePath + "3.txt")

assert data1.shape == data2.shape
assert width1 == width2 and height1 == height2

# Grid
x = np.linspace(0, width1, data1.shape[1])
y = np.linspace(0, height1, data1.shape[0])
X, Y = np.meshgrid(x, y)

# Convert to nanometers
data1_nm = data1 * 1e9
data2_nm = data2 * 1e9

vmin = min(data1_nm.min(), data2_nm.min())
vmax = max(data1_nm.max(), data2_nm.max())

# Set up figure with GridSpec: 2 rows, 2 columns (last column for colorbar)
fig = plt.figure(figsize=(10, 14))
gs = gridspec.GridSpec(
    2, 2, 
    width_ratios=[38, 1],  # Wider plots, thinner colorbar
    height_ratios=[1, 1], 
    wspace=0,           # Smaller gap between plots and colorbar
    hspace=0.15
)


# Subplot for 1.txt
ax1 = fig.add_subplot(gs[0, 0])
cf1 = ax1.contourf(X, Y, data1_nm, levels=50, vmin=vmin, vmax=vmax, cmap='viridis')
ax1.set_title("Topography natural surfaces")
ax1.set_ylabel("Y (μm)")
ax1.set_aspect('equal')

# Subplot for 3.txt
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
cf2 = ax2.contourf(X, Y, data2_nm, levels=50, vmin=vmin, vmax=vmax, cmap='viridis')
# ax2.set_title("Topography: 3.txt")
ax2.set_xlabel("X (μm)")
ax2.set_ylabel("Y (μm)")
ax2.set_aspect('equal')

# Shared colorbar on the right, spanning both rows
# Manually add colorbar: [left, bottom, width, height]
cbar_ax = fig.add_axes([0.8, 0.11, 0.02, 0.77])  # Adjust left (0.88) to move closer
cb = fig.colorbar(cf2, cax=cbar_ax)
cb.set_label("Height (nm)")


plt.savefig("topography_comparison.png", dpi=300, bbox_inches='tight')
