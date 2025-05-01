import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

filenames = ["top", "bottom", "left", "right", "front", "back"]


# Plot the data
fig, axs = plt.subplots(3, 2, figsize = (12,10))

for i, name in enumerate(filenames):
    data = pd.read_csv('FilesCSV/powers_polarization_' + name + '.csv', header=None)

    col = i % 2
    row = i // 2

    axs[row, col].plot(data[0], marker='o', linestyle='-', color='b')
    axs[row, col].set_title("Power Values " + name)
    axs[row, col].grid(True)


for ax in axs.flat:
    ax.set(xlabel='Index', ylabel='Value')


fig.tight_layout()
plt.show()

# Plot the data
fig, axs = plt.subplots(3, 2, figsize = (12,10))

for i, name in enumerate(filenames):
    data = pd.read_csv('FilesCSV/integrand_beta0_' + name + '.csv', header=None)

    col = i % 2
    row = i // 2

    im = axs[row, col].imshow(data,cmap='viridis',interpolation='nearest')
    axs[row, col].set_title("Integrand (beta = 0) " + name)
    axs[row, col].grid(True)
    plt.colorbar(im, ax = axs[row, col])



fig.tight_layout()
plt.show()