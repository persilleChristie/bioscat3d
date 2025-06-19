import os
import matplotlib.pyplot as plt
from plotUtils import load_error_summary_df, plot_error_vector_from_file
import seaborn as sns
import matplotlib.cm as cm
from plotUtils import plot_stat_vs_variable

# ------------------------------
# Configuration
# ------------------------------
folder_read = "../../CSV/TangentialErrorsEFixedUnitsFromTen"
folder_out = "TangentialErrorsEFixedUnitsFromTen"
df = load_error_summary_df(folder_read)

# Style config: consistent across all plots
unique_lambdas = sorted(df["lam"].unique())
#lambda_colors = plt.colormaps.get_cmap("tab10", len(unique_lambdas))
lambda_colors = plt.colormaps.get_cmap("tab10")

#lambda_to_color = {lam: lambda_colors(i) for i, lam in enumerate(unique_lambdas)}
lambda_to_color = {lam: lambda_colors(i % lambda_colors.N) for i, lam in enumerate(unique_lambdas)}
tau_to_linestyle = {"tau1": "-", "tau2": "--"}

# ------------------------------
# Task 1: All τ₁ and τ₂ error vectors (individual and grid)
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

        plot_error_vector_from_file(
            filepath, ax=ax, linestyle=tau_to_linestyle[tau]
        )

        lam = row["lam"]
        beta = row["beta"]
        auxpts = row["auxpts"]
        ax.set_title(f"{tau}: λ={lam:.3f}, β={beta:.3f}, auxpts={auxpts}", fontsize=10)
        ax.set_xlabel("Point Index")
        ax.set_ylabel("Error Magnitude")
        ax.grid(True)

        outname = row["filename"].replace(".csv", ".png")
        fig.tight_layout()
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

        plot_error_vector_from_file(
            filepath, ax=ax, linestyle=tau_to_linestyle[tau]
        )

        lam = row["lam"]
        beta = row["beta"]
        auxpts = row["auxpts"]
        ax.set_title(f"λ={lam:.3f}, β={beta:.3f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove unused subplots
    for idx in range(len(df_tau), nrows * ncols):
        r, c = divmod(idx, ncols)
        fig.delaxes(axes[r][c])

    fig.suptitle(f"All raw error vectors for {tau}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(folder_grid, f"{tau}_grid.png"), dpi=150)
    plt.close(fig)


# ------------------------------
# Task 2: max_error vs β for each λ (individual + grid)
# ------------------------------
from plotUtils import plot_stat_vs_variable

summary_folder = os.path.join(folder_out, "summary")
os.makedirs(summary_folder, exist_ok=True)

metric = "max_error"     # Default as per user preference
aux_fixed = 5            # Fixed number of auxiliary points to filter by
ncols = 2
dpi = 150

folder_beta = os.path.join(summary_folder, f"{metric}_vs_beta")
os.makedirs(folder_beta, exist_ok=True)

lambdas = sorted(df["lam"].unique())
nrows = (len(lambdas) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

for idx, lam in enumerate(lambdas):
    df_lam = df[(df["lam"] == lam) & (df["auxpts"] == aux_fixed)]

    # --- Individual Plot ---
    fig_indiv, ax = plt.subplots()
    for tau in ["tau1", "tau2"]:
        df_sub = df_lam[df_lam["tau_type"] == tau]
        if not df_sub.empty:
            ax.plot(
                df_sub["beta"],
                df_sub[metric],
                label=f"{tau}",
                linestyle=tau_to_linestyle[tau],
                color=lambda_to_color[lam],
                marker="o"
            )
    ax.set_title(f"{metric} vs β (λ = {lam:.3f})")
    ax.set_xlabel("β")
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend()
    fig_indiv.tight_layout()
    fig_indiv.savefig(os.path.join(folder_beta, f"lam{int(1000*lam):03d}.png"), dpi=dpi)
    plt.close(fig_indiv)

    # --- Grid Plot ---
    r, c = divmod(idx, ncols)
    ax_grid = axes[r][c]
    for tau in ["tau1", "tau2"]:
        df_sub = df_lam[df_lam["tau_type"] == tau]
        if not df_sub.empty:
            ax_grid.plot(
                df_sub["beta"],
                df_sub[metric],
                label=f"{tau}",
                linestyle=tau_to_linestyle[tau],
                color=lambda_to_color[lam],
                marker="o"
            )
    ax_grid.set_title(f"λ = {lam:.3f}")
    ax_grid.set_xlabel("β")
    ax_grid.set_ylabel(metric)
    ax_grid.grid(True)
    ax_grid.legend(fontsize=8)

# Remove empty subplots
for i in range(len(lambdas), nrows * ncols):
    r, c = divmod(i, ncols)
    fig.delaxes(axes[r][c])

fig.suptitle(f"{metric} vs β (fixed auxpts = {aux_fixed})", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(folder_beta, "all_lambdas_grid.png"), dpi=dpi)
plt.close(fig)



# ------------------------------
# Task 3: Max error vs λ for each fixed β (with all τ₁ and τ₂ in same plots)
# ------------------------------
folder_summary = os.path.join(folder_out, "summary", "max_error_vs_lambda")
folder_indiv = os.path.join(folder_summary, "individual")
os.makedirs(folder_indiv, exist_ok=True)

# Filter only fixed auxpts
df_fixed_aux = df[df["auxpts"] == aux_fixed]
unique_betas = sorted(df_fixed_aux["beta"].unique())

# --- Individual plots ---
for beta in unique_betas:
    df_beta = df_fixed_aux[df_fixed_aux["beta"] == beta]
    fig, ax = plt.subplots()

    for tau in ["tau1", "tau2"]:
        df_tau = df_beta[df_beta["tau_type"] == tau]
        ax.plot(
            df_tau["lam"],
            df_tau["max_error"],
            label=f"{tau}",
            linestyle=tau_to_linestyle[tau],
            marker="o"
        )

    ax.set_title(f"Max error vs λ for β = {beta:.3f}")
    ax.set_xlabel("λ")
    ax.set_ylabel("Max Error")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(folder_indiv, f"beta{int(beta*1000):03d}.png"), dpi=150)
    plt.close(fig)

# --- Grid of subplots ---
n = len(unique_betas)
ncols = 2
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

for idx, beta in enumerate(unique_betas):
    r, c = divmod(idx, ncols)
    ax = axes[r][c]
    df_beta = df_fixed_aux[df_fixed_aux["beta"] == beta]

    for tau in ["tau1", "tau2"]:
        df_tau = df_beta[df_beta["tau_type"] == tau]
        ax.plot(
            df_tau["lam"],
            df_tau["max_error"],
            label=f"{tau}",
            linestyle=tau_to_linestyle[tau],
            marker="o"
        )

    ax.set_title(f"β = {beta:.3f}")
    ax.set_xlabel("λ")
    ax.set_ylabel("Max Error")
    ax.grid(True)
    ax.legend(fontsize=8)

# Remove empty subplots
for idx in range(n, nrows * ncols):
    r, c = divmod(idx, ncols)
    fig.delaxes(axes[r][c])

fig.suptitle("Max error vs λ for each β (auxpts = 5)", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(folder_summary, "all_betas_grid.png"), dpi=150)
plt.close(fig)


# ------------------------------
# Task 4: max_error vs auxpts for each fixed β (all λ curves)
# ------------------------------
folder_summary = os.path.join(folder_out, "summary", "max_error_vs_auxpts_by_beta")
folder_indiv = os.path.join(folder_summary, "individual")
os.makedirs(folder_indiv, exist_ok=True)

unique_betas = sorted(df["beta"].unique())

for beta in unique_betas:
    df_beta = df[df["beta"] == beta]
    fig, ax = plt.subplots()

    for lam in unique_lambdas:
        for tau in ["tau1", "tau2"]:
            df_plot = df_beta[(df_beta["lam"] == lam) & (df_beta["tau_type"] == tau)]
            if not df_plot.empty:
                ax.plot(
                    df_plot["auxpts"],
                    df_plot["max_error"],
                    label=f"λ = {lam:.3f}, {tau}",
                    linestyle=tau_to_linestyle[tau],
                    color=lambda_to_color[lam],
                    marker="o"
                )

    ax.set_title(f"Max error vs auxpts for β = {beta:.3f}")
    ax.set_xlabel("Auxiliary Points")
    ax.set_ylabel("Max Error")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(folder_indiv, f"beta{int(beta*1000):03d}.png"), dpi=150)
    plt.close(fig)



# ------------------------------
# Task 5: Heatmap of max error across (λ, β) for fixed auxpts
# ------------------------------

folder_heatmap = os.path.join(folder_out, "summary", "heatmaps")
os.makedirs(folder_heatmap, exist_ok=True)

# Only fixed auxpts
df_heatmap = df[df["auxpts"] == aux_fixed]

for tau in ["tau1", "tau2"]:
    df_tau = df_heatmap[df_heatmap["tau_type"] == tau]

    # Pivot table: rows = lambda, cols = beta, values = max_error
    heatmap_data = df_tau.pivot(index="lam", columns="beta", values="max_error")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2e", cmap="viridis", ax=ax)

    ax.set_title(f"Heatmap of max error (auxpts = {aux_fixed}, {tau})", fontsize=14)
    ax.set_xlabel("β")
    ax.set_ylabel("λ")

    # Save
    fname = f"heatmap_max_error_{tau}_auxpts{aux_fixed}.png"
    fig.tight_layout()
    fig.savefig(os.path.join(folder_heatmap, fname), dpi=150)
    plt.close(fig)
