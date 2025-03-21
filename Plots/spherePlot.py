import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('..\\E_field_unit_sphere.csv')

# Extract coordinates and magnitude
x, y, z, E_mag = data['x'], data['y'], data['z'], data['E_mag']

# 3D scatter plot
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter(x, y, z, c=E_mag, cmap='viridis', marker='o', s=15)

# Adding a colorbar
cbar = plt.colorbar(scat, shrink=0.5, pad=0.1)
cbar.set_label('|E_tot| magnitude', fontsize=12)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Electric Field Magnitude on Unit Sphere')

plt.show()
