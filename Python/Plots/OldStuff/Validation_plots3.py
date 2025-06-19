import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os
from collections import defaultdict

folder = "../../CSV/TangentialErrorsE/"
pattern1 = os.path.join(folder, "E_tangential_error1_auxpts5_*.csv")
pattern2 = os.path.join(folder, "E_tangential_error2_auxpts5_*.csv")
regex = r"E_tangential_error[12]_auxpts5_lambda(\d+)_beta(\d+)\.csv"

# ----------- Load and Organize Data -------------
errors = defaultdict(dict)  # (lambda, beta) → {1: series, 2: series}

for pattern in [pattern1, pattern2]:
    for file in glob.glob(pattern):
        match = re.search(regex, os.path.basename(file))
        if match:
            lam = int(match.group(1)) / 1000
            beta = int(match.group(2)) / 1000
            kind = 1 if "error1" in file else 2
            data = pd.read_csv(file, header=None).squeeze("columns")
            errors[(lam, beta)][kind] = data

# ----------- Group by beta -------------
grouped_by_beta = defaultdict(dict)
for (lam, beta), d in errors.items():
    if 1 in d and 2 in d:
        grouped_by_beta[beta][lam] = d

# ----------- Plot by beta -------------
for beta, lam_dict in sorted(grouped_by_beta.items()):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    for lam in sorted(lam_dict.keys()):
        err1 = lam_dict[lam][1]
        err2 = lam_dict[lam][2]
        err_sum = err1 + err2

        axes[0].plot(err1, label=f"λ={lam:.3f}")
        axes[1].plot(err2, label=f"λ={lam:.3f}")
        axes[2].plot(err_sum, label=f"λ={lam:.3f}")

    axes[0].set_title("Tangential Error 1")
    axes[1].set_title("Tangential Error 2")
    axes[2].set_title("Sum of Errors")

    for ax in axes:
        ax.set_xlabel("Point index")
        ax.set_ylabel("Error")
        ax.grid(True)
        ax.legend()

    plt.suptitle(f"Tangential Errors (auxpts=5, β={beta:.3f})")
    plt.tight_layout()
    plt.savefig(f"ResultsFixedRadius/subplot_Et1Et2Sum_auxpts5_beta{beta:.3f}.png")
    plt.close()

# ----------- Group by lambda -------------
grouped_by_lambda = defaultdict(dict)
for (lam, beta), d in errors.items():
    if 1 in d and 2 in d:
        grouped_by_lambda[lam][beta] = d

# ----------- Plot by lambda -------------
for lam, beta_dict in sorted(grouped_by_lambda.items()):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    for beta in sorted(beta_dict.keys()):
        err1 = beta_dict[beta][1]
        err2 = beta_dict[beta][2]
        err_sum = err1 + err2

        axes[0].plot(err1, label=f"β={beta:.3f}")
        axes[1].plot(err2, label=f"β={beta:.3f}")
        axes[2].plot(err_sum, label=f"β={beta:.3f}")

    axes[0].set_title("Tangential Error 1")
    axes[1].set_title("Tangential Error 2")
    axes[2].set_title("Sum of Errors")

    for ax in axes:
        ax.set_xlabel("Point index")
        ax.set_ylabel("Error")
        ax.grid(True)
        ax.legend()

    plt.suptitle(f"Tangential Errors (auxpts=5, λ={lam:.3f})")
    plt.tight_layout()
    plt.savefig(f"ResultsFixedRadius/subplot_Et1Et2Sum_auxpts5_lambda{lam:.3f}.png")
    plt.close()
