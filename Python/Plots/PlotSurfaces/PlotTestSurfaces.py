import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

keyword = "NormalNewGeom"

jsonpath = "../../../json/surfaceParams" + keyword + ".json"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
    "figure.titlesize": 22
})

# Load JSON data
with open(jsonpath, "r") as f:
    data = json.load(f)

# Extract bump data
bump_data = data["bumpData"]

# Define the domain from the half-widths
x_min, x_max = -data["halfWidth_x"], data["halfWidth_x"]
y_min, y_max = -data["halfWidth_y"], data["halfWidth_y"]

# Resolution (points per unit length)
res = 1000
nx = int((x_max - x_min) * res)
ny = int((y_max - y_min) * res)

x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Evaluate the sum of all Gaussian bumps
Z = np.zeros_like(X)
for bump in bump_data:
    x0 = bump["x0"]
    y0 = bump["y0"]
    height = bump["height"]
    sigma = bump["sigma"]

    exponent = - ((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)
    Z += height * np.exp(exponent)
Z *= 1e3

# Plot
# plt.figure(figsize=(6, 5))
# contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
# plt.colorbar(contour, label=r'Height (nm)')
# plt.xlabel(r'X ($\mu$m)')
# plt.ylabel(r'Y ($\mu$m)')
# plt.title(r'Contour $\xi_{0.5}$')

# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# # Ensure same number of ticks on X and Y
# xticks = np.linspace(x_min, x_max, num=5)  # Choose 7 ticks (for example)
# yticks = np.linspace(y_min, y_max, num=5)  # Same number as X

# ax.set_xticks(xticks)
# ax.set_yticks(yticks)

# plt.tight_layout()
# plt.savefig("Surface" + keyword + "Plot.png", dpi=300)



plt.figure(figsize=(6, 5))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')

# Remove axes
plt.axis('off')

# Remove spines and ticks
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Optional: Make background transparent or white
plt.gca().patch.set_alpha(0)  # Transparent background inside the axes
plt.gcf().patch.set_alpha(0)  # Transparent background outside the axes

plt.tight_layout(pad=0)
plt.savefig("Surface" + keyword + "Plot_clean.png", dpi=1500, bbox_inches='tight', pad_inches=0, transparent=True)

