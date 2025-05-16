import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load system matrix
df = pd.read_csv('FilesCSV/matrix_A_simple.csv', sep=",", header=None)

df = df.astype('string')

# Calculate condition number and number of zeros
A = df.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()
_, S, _ = np.linalg.svd(A)
print(f"Condition number PN: {S[0]/S[-1]}")

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
im0 = axs[0].imshow(abs(A), cmap='viridis')
axs[0].set_title("PN system matrix")
plt.colorbar(im0, ax=axs[0])



for i in range(7):
    data = pd.read_csv('FilesCSV/vector_b_' + str(i) + '.csv', header=None)
    data = data.astype('string')

    b = data.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()

    if i == 0:
        bMat = abs(b)
    else:
        bMat = np.append(bMat, abs(b), axis = 1)

im1 = axs[1].imshow(bMat, cmap='viridis')
axs[1].set_title("PN abs rhs")
plt.colorbar(im1, ax=axs[1])

# data = pd.read_csv('FilesCSV/vector_b_0.csv', header=None)
# data = data.astype('string')

# b = data.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()

# plt.plot(abs(b), color='b')
# # axs[row, col].plot(abs(y_A[:,i]), color='r')
# plt.title("Abs rhs, first polarization")
# plt.grid(True)


# for ax in axs.flat:
#     ax.set(xlabel='Index', ylabel='Value')


# df = pd.read_csv('FilesCSV/Andreas_y_matrix.csv', header=None)

# y_A = df.applymap(complex).to_numpy()

# fig, axs = plt.subplots(5, 2, figsize = (12,10))



for i in range(7):
    data = pd.read_csv('FilesCSV/solution_y_' + str(i) + '.csv', header=None)
    data = data.astype('string')

    y = data.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()

    if i == 0:
        yMat = abs(y)
    else:
        yMat = np.append(yMat, abs(y), axis = 1)

    # col = i % 2
    # row = i // 2

    # axs[row, col].plot(abs(y), color='b')
    # # axs[row, col].plot(abs(y_A[:,i]), color='r')
    # axs[row, col].set_title("Abs solution, beta = {:.2f}".format(betas[i]))
    # axs[row, col].grid(True)

im2 = axs[2].imshow(yMat, cmap='viridis')
axs[2].set_title("PN abs solution")
plt.colorbar(im2, ax=axs[2])

# data = pd.read_csv('FilesCSV/solution_y_0.csv', header=None)
# data = data.astype('string')

# y = data.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()

# plt.plot(abs(y), color='b')
# # axs[row, col].plot(abs(y_A[:,i]), color='r')
# plt.title("Abs solution, first polarization")
# plt.grid(True)


# for ax in axs.flat:
#     ax.set(xlabel='Index', ylabel='Value')


# fig.tight_layout()
# plt.show()


fig.tight_layout()
plt.show()