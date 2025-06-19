import numpy as np
import matplotlib.pyplot as plt

# Define the Gaussian
x_vals = np.linspace(-5, 5, 200)
A = 2 / np.sqrt(2 * np.pi)
f_vals = A * np.exp(-x_vals**2 / 2)

# Choose normal vector points
x_norm = np.array([-4, -2, 0, 2, 4])
f_norm = A * np.exp(-x_norm**2 / 2)
df_norm = -x_norm * f_norm  # correct derivative

# Compute actual normal vectors (rotate tangent)
arrow_length = 0.5  # increase to make them visible
normals = []
for x, f, df in zip(x_norm, f_norm, df_norm):
    norm = np.sqrt(df**2 + 1)
    dx = -df / norm
    dy = 1 / norm
    x2 = x + arrow_length * dx
    y2 = f + arrow_length * dy
    normals.append(((x, f), (x2, y2)))

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_vals, f_vals, color='blue', linewidth=2)

for (x1, y1), (x2, y2) in normals:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

ax.axis('off')
ax.set_xlim(-5, 5)
ax.set_ylim(-0.05, 0.45)
plt.tight_layout()
plt.show()
