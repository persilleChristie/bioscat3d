import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os

# Adjust path as needed
folder = "../../CSV/TangentialErrorsE/"
pattern = os.path.join(folder, "E_tangential_error1_*.csv")

# Match parts of the filename
regex = r"E_tangential_error1_auxpts(\d+)_lambda(\d+)_beta(\d+)\.csv"

for file in glob.glob(pattern):
    match = re.search(regex, os.path.basename(file))
    if match:
        auxpts = int(match.group(1))
        lam = int(match.group(2)) / 1000  # undo the *1000 scaling
        beta = int(match.group(3)) / 1000

        data = pd.read_csv(file, header=None).squeeze("columns")

        plt.figure(figsize=(8, 4))
        plt.plot(data)
        plt.title(f"$E^1_{{tangential}}$: auxpts={auxpts}, λ={lam:.3f}, β={beta:.3f}")
        plt.xlabel("Point index")
        plt.ylabel("Error magnitude")
        plt.grid(True)
        plt.tight_layout()

        # Optional: save each plot
        outname = f"ResultsFixedRadius/plot_Et1_auxpts{auxpts}_lambda{lam:.3f}_beta{beta:.3f}.png"
        plt.savefig(outname)

        # Or display directly
        # plt.show()
        plt.close()
