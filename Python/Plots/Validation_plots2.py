import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os
from collections import defaultdict

# ------------ Settings ------------
auxpts = 5  # change to 4, 6, etc. if needed
folder = f"../../CSV/TangentialErrorsE/"
pattern = os.path.join(folder, f"E_tangential_error1_auxpts{auxpts}_*.csv")
regex = rf"E_tangential_error1_auxpts{auxpts}_lambda(\d+)_beta(\d+)\.csv"

# ------------ Load files ------------
by_lambda = defaultdict(dict)  # lambda → beta → Series
by_beta = defaultdict(dict)    # beta → lambda → Series

files = glob.glob(pattern)
if not files:
    raise RuntimeError(f"No files found matching pattern: {pattern}")

for file in files:
    match = re.search(regex, os.path.basename(file))
    if match:
        lam = int(match.group(1)) / 1000
        beta = int(match.group(2)) / 1000
        data = pd.read_csv(file, header=None).squeeze("columns")
        by_lambda[lam][beta] = data
        by_beta[beta][lam] = data
    else:
        print(f"Skipped file (no match): {file}")

# ------------ Plot: Fixed beta, varying lambda ------------
if not by_lambda:
    raise RuntimeError("No valid files matched the regex. Check regex or filenames.")

fixed_beta = sorted(by_lambda[next(iter(by_lambda))].keys())[0]
plt.figure(figsize=(10, 6))
for lam in sorted(by_lambda.keys()):
    if fixed_beta in by_lambda[lam]:
        series = by_lambda[lam][fixed_beta]
        plt.plot(series, label=f"λ={lam:.3f}")
plt.title(f"$E^1_{{tangential}}$ for auxpts={auxpts}, β={fixed_beta:.3f} (varying λ)")
plt.xlabel("Point index")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"Et1_auxpts{auxpts}_by_lambda.png")
plt.close()

# ------------ Plot: Fixed lambda, varying beta ------------
fixed_lambda = sorted(by_beta[next(iter(by_beta))].keys())[0]
plt.figure(figsize=(10, 6))
for beta in sorted(by_beta.keys()):
    if fixed_lambda in by_beta[beta]:
        series = by_beta[beta][fixed_lambda]
        plt.plot(series, label=f"β={beta:.3f}")
plt.title(f"$E^1_{{tangential}}$ for auxpts={auxpts}, λ={fixed_lambda:.3f} (varying β)")
plt.xlabel("Point index")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"ResultsFixedRadius/Et1_auxpts{auxpts}_by_beta.png")
plt.close()
