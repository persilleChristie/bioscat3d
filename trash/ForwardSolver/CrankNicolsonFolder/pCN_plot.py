import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# For a 1D vector, you’ll get a single-row DataFrame
df = pd.read_csv("totalcount.csv", header=None)

# Flatten to 1D array (row → series)
data = df.values.flatten()

# Plot
plt.plot(data, marker='o')
plt.show()