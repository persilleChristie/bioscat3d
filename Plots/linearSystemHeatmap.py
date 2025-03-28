import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../matrix_A.csv', sep=",", header=None)

df = df.astype('string')

A = df.applymap(lambda x: complex(x.replace("i", "j").replace("+-","-"))).to_numpy()

plt.imshow(abs(A), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
