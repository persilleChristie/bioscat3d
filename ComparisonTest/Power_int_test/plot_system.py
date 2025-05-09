import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

betas = [
        0.0,
        0.17453292519943295,
        0.3490658503988659,
        0.5235987755982988,
        0.6981317007977318,
        0.8726646259971648,
        1.0471975511965976,
        1.2217304763960306,
        1.3962634015954636,
        1.5707963267948966
    ]

# df = pd.read_csv('FilesCSV/Andreas_y_matrix.csv', header=None)

# y_A = df.applymap(complex).to_numpy()

# fig, axs = plt.subplots(5, 2, figsize = (12,10))

# for i in range(10):
#     data = pd.read_csv('FilesCSV/solution_y_' + str(i) + '.csv', header=None)
#     data = data.astype('string')

#     y = data.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()

#     col = i % 2
#     row = i // 2

#     axs[row, col].plot(abs(y), color='b')
#     # axs[row, col].plot(abs(y_A[:,i]), color='r')
#     axs[row, col].set_title("Abs solution, beta = {:.2f}".format(betas[i]))
#     axs[row, col].grid(True)

data = pd.read_csv('FilesCSV/solution_y_0.csv', header=None)
data = data.astype('string')

y = data.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()

plt.plot(abs(y), color='b')
# axs[row, col].plot(abs(y_A[:,i]), color='r')
plt.title("Abs solution, first polarization")
plt.grid(True)


# for ax in axs.flat:
#     ax.set(xlabel='Index', ylabel='Value')


# fig.tight_layout()
plt.show()


data = pd.read_csv('FilesCSV/vector_b_0.csv', header=None)
data = data.astype('string')

b = data.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()

plt.plot(abs(b), color='b')
# axs[row, col].plot(abs(y_A[:,i]), color='r')
plt.title("Abs rhs, first polarization")
plt.grid(True)


# for ax in axs.flat:
#     ax.set(xlabel='Index', ylabel='Value')


# fig.tight_layout()
plt.show()