import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filenames = ["top", "bottom", "left", "right", "front", "back"]


# Plot the data
fig, axs = plt.subplots(3, 2, figsize = (12,10))

for i, name in enumerate(filenames):
    data = pd.read_csv('FilesCSV/powers_polarization_' + name + '.csv', header=None)

    betas = np.linspace(0, 90, len(data))

    col = i % 2
    row = i // 2

    axs[row, col].plot(betas, data[0], marker='o', linestyle='-', color='b')
    axs[row, col].set_title("Power Values " + name)
    axs[row, col].grid(True)


for ax in axs.flat:
    ax.set(xlabel='Polarization angle', ylabel='Value')


fig.tight_layout()
plt.show()


# x = np.linspace(-2, 2, 20)
# y = np.linspace(-2, 2, 20)

# X, Y = np.meshgrid(x, y)

# # Plot the data
# fig, axs = plt.subplots(3, 2, figsize = (12,10))

# for i, name in enumerate(filenames):
#     data = pd.read_csv('FilesCSV/integrand_beta0_' + name + '.csv', header=None)

#     col = i % 2
#     row = i // 2

#     # im = axs[row, col].imshow(data,cmap='viridis',interpolation='nearest')
#     axs[row, col].contourf(X, Y, data)
#     axs[row, col].set_title("Integrand (beta = 0) " + name)
#     axs[row, col].grid(True)
#     # axs[row, col].invert_yaxis()
#     # plt.colorbar(im, ax = axs[row, col])



# fig.tight_layout()
# plt.show()