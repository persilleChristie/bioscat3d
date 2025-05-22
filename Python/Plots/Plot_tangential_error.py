import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv("../../CSV/tangential_error1.csv", header = None)
df2 = pd.read_csv("../../CSV/tangential_error2.csv", header = None)

error1 = df1.to_numpy()
error2 = df2.to_numpy()

figure, ax = plt.subplots()
ax.plot(error1, label = "Tangent 1")
ax.plot(error2, label = "Tangent 2")
ax.legend()

plt.show()