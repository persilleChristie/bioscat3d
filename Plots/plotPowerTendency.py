import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file (assuming no header)
data = pd.read_csv('../ForwardSolver/FilesCSV/powers_polarization.csv', header=None)

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(data[0], marker='o', linestyle='-', color='b')
plt.title("Line Plot of Power Values")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.show()
