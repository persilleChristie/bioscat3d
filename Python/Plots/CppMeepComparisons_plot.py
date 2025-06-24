import os
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
    "figure.titlesize": 22
})

# --- Configuration ---
csv_dir = "../../CSV/MeepFlux"
file_pattern = os.path.join(csv_dir, "Flux_*.csv")
num_betas = 7  # Set this to your actual beta resolution
beta_vals = np.linspace(0, np.pi / 2, num_betas)


# --- Helper: Convert 'tag' back to float approximation ---
def untag(tag):
    return float(tag[:3]) / 100

# --- Parse all files ---
files = glob(file_pattern)

print(f"Found {len(files)} files matching pattern '{file_pattern}'")
print("Files:", files)

surface_data = {}

for filepath in files:
    filename = os.path.basename(filepath)
    match = re.match(r"Flux_(\w+)_lambdamin_(\d+)_lambdamax_(\d+)\.csv", filename)
    if not match:
        continue
    surface, lam_min_tag, lam_max_tag = match.groups()
    lam_min = untag(lam_min_tag)
    lam_max = untag(lam_max_tag)
    
    matrix = np.loadtxt(filepath, delimiter=',')
    
    num_lambdas = matrix.shape[0]
    lambda_vals = np.linspace(lam_min, lam_max, num_lambdas)

    surface_data[surface] = {
        "lambda_vals": lambda_vals,
        "beta_vals": beta_vals,
        "flux": matrix
    }

# --- Plotting ---
output_dir = "./CppMeepComparisons"
os.makedirs(output_dir, exist_ok=True)

for surface, data in surface_data.items():
    λ = data["lambda_vals"]
    β = data["beta_vals"]
    flux = data["flux"]

    # Plot flux vs. lambda for each beta
    plt.figure(figsize=(8, 6))
    for i in range(flux.shape[1]):
        plt.plot(λ, flux[:, i], label=f"β={β[i]:.2f}")
    plt.title(f"Scattered flux vs. Frequency - Direction {surface}")
    plt.xlabel("Wavelength (λ)")
    #plt.ylabel("Flux")
    plt.legend(title="Polarisation Angle")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{surface}_flux_vs_lambda.png")
    plt.close()

    # Plot flux vs. beta for each lambda
    plt.figure(figsize=(8, 6))
    for i in range(flux.shape[0]):
        plt.plot(β, flux[i, :], label=f"λ={λ[i]:.3f}")
    plt.title(f"Scattered flux vs. Polarisation - Direction {surface}")
    plt.xlabel("Beta (β)")
    #plt.ylabel("Flux")
    plt.legend(title="Wavelength")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{surface}_flux_vs_beta.png")
    plt.close()

print(f"Plots saved in {output_dir}/")
