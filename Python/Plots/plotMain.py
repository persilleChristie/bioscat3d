import os
import matplotlib.pyplot as plt
from plotUtils import load_error_summary_df, plot_error_vector_from_file

# ------------------------------
# Configuration
# ------------------------------
folder_read = "../../CSV/TangentialErrorsECurvNoGuard"
folder_out = "TangentialErrorsECurvNoGuard"
df = load_error_summary_df(folder_read)

# ------------------------------
# Plotting Task 1: All τ₁ and τ₂ error vectors
# ------------------------------
for tau in ["tau1", "tau2"]:
    df_tau = df[df["tau_type"] == tau]
    
    folder_individual = os.path.join(folder_out, tau, "individual")
    folder_grid = os.path.join(folder_out, tau, "grid")
    os.makedirs(folder_individual, exist_ok=True)
    os.makedirs(folder_grid, exist_ok=True)

    # --- 1. Individual plots ---
    for _, row in df_tau.iterrows():
        fig, ax = plt.subplots()
        filepath = os.path.join(folder_read, row["filename"])
        plot_error_vector_from_file(filepath, ax=ax)
        outname = row["filename"].replace(".csv", ".png")
        fig.savefig(os.path.join(folder_individual, outname), dpi=150)
        plt.close(fig)

    # --- 2. Grid of subplots ---
    n = len(df_tau)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, (_, row) in enumerate(df_tau.iterrows()):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        filepath = os.path.join(folder_read, row["filename"])
        plot_error_vector_from_file(filepath, ax=ax)

    # Remove empty subplots
    for idx in range(len(df_tau), nrows * ncols):
        r, c = divmod(idx, ncols)
        fig.delaxes(axes[r][c])

    fig.suptitle(f"All raw error vectors for {tau}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(folder_grid, f"{tau}_grid.png"), dpi=150)
    plt.close(fig)
from plotUtils import plot_stat_vs_variable

# ------------------------------
# Plotting Task 2: Summary plots of mean_error (flat output structure)
# ------------------------------
from plotUtils import plot_stat_vs_variable

summary_root = os.path.join(folder_out, "summary")
metric = "mean_error"

# --- 1. Plot vs β for each λ individually + subplot grid ---
folder_beta = os.path.join(summary_root, f"{metric}_vs_beta")
os.makedirs(folder_beta, exist_ok=True)
lambdas = sorted(df["lam"].unique())

ncols = 3
nrows = (len(lambdas) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

for idx, lam in enumerate(lambdas):
    df_lam = df[df["lam"] == lam]
    fig_indiv, ax = plt.subplots()
    save_path = os.path.join(folder_beta, f"lam{int(1000*lam):03d}.png")
    plot_stat_vs_variable(df_lam, x="beta", y=metric, hue="tau_type", ax=ax,
                          title=f"{metric} vs β (λ = {lam:.3f})", save_path=save_path)
    plt.close(fig_indiv)

    r, c = divmod(idx, ncols)
    plot_stat_vs_variable(df_lam, x="beta", y=metric, hue="tau_type", ax=axes[r][c],
                          title=f"λ = {lam:.3f}")

for i in range(len(lambdas), nrows*ncols):
    r, c = divmod(i, ncols)
    fig.delaxes(axes[r][c])

fig.suptitle(f"{metric} vs β — all λ", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(folder_beta, "all_lambdas_grid.png"), dpi=150)
plt.close(fig)

# --- 2. Plot vs λ for each β individually + subplot grid ---
folder_lambda = os.path.join(summary_root, f"{metric}_vs_lambda")
os.makedirs(folder_lambda, exist_ok=True)
betas = sorted(df["beta"].unique())

ncols = 3
nrows = (len(betas) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

for idx, beta in enumerate(betas):
    df_beta = df[df["beta"] == beta]
    fig_indiv, ax = plt.subplots()
    save_path = os.path.join(folder_lambda, f"beta{int(1000*beta):03d}.png")
    plot_stat_vs_variable(df_beta, x="lam", y=metric, hue="tau_type", ax=ax,
                          title=f"{metric} vs λ (β = {beta:.3f})", save_path=save_path)
    plt.close(fig_indiv)

    r, c = divmod(idx, ncols)
    plot_stat_vs_variable(df_beta, x="lam", y=metric, hue="tau_type", ax=axes[r][c],
                          title=f"β = {beta:.3f}")

for i in range(len(betas), nrows*ncols):
    r, c = divmod(i, ncols)
    fig.delaxes(axes[r][c])

fig.suptitle(f"{metric} vs λ — all β", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(folder_lambda, "all_betas_grid.png"), dpi=150)
plt.close(fig)

# --- 3. Plot vs auxpts for each λ (individual only) ---
folder_aux_lam = os.path.join(summary_root, f"{metric}_vs_auxpts")
os.makedirs(folder_aux_lam, exist_ok=True)
for lam in lambdas:
    df_lam = df[df["lam"] == lam]
    save_path = os.path.join(folder_aux_lam, f"lam{int(1000*lam):03d}.png")
    plot_stat_vs_variable(df_lam, x="auxpts", y=metric, hue="tau_type",
                          title=f"{metric} vs auxpts (λ = {lam:.3f})", save_path=save_path)

# --- 4. Plot vs auxpts for each β (individual only) ---
for beta in betas:
    df_beta = df[df["beta"] == beta]
    save_path = os.path.join(folder_aux_lam, f"beta{int(1000*beta):03d}.png")
    plot_stat_vs_variable(df_beta, x="auxpts", y=metric, hue="tau_type",
                          title=f"{metric} vs auxpts (β = {beta:.3f})", save_path=save_path)
