import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd
from plotUtils import load_error_summary_df, plot_error_vector_from_file, plot_stat_vs_variable
from matplotlib.gridspec import GridSpec

# Font settings (already applied globally in full script earlier)
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
    "figure.titlesize": 20
})

# ------------------------------
# Configuration
# ------------------------------
folder_read = "../../CSV/TangentialErrorsEFixedUnits10FromTen"
folder_out = "TangentialErrorsEFixedUnits10FromTenDynamic"
df = load_error_summary_df(folder_read)

unique_lambdas = sorted(df["lam"].unique())
lambda_colors = plt.colormaps.get_cmap("tab10")
lambda_to_color = {lam: lambda_colors(i % lambda_colors.N) for i, lam in enumerate(unique_lambdas)}
tau_to_linestyle = {"tau1": "-", "tau2": "--"}

# ------------------------------
# Task 1: All τ₁ and τ₂ error vectors
# ------------------------------
tau_labels = {"tau1": r"$\tau_1$", "tau2": r"$\tau_2$"}

for tau in ["tau1", "tau2"]:
    df_tau = df[df["tau_type"] == tau]
    tau_label = tau_labels[tau]

    folder_individual = os.path.join(folder_out, tau, "individual")
    folder_grid = os.path.join(folder_out, tau, "grid")
    os.makedirs(folder_individual, exist_ok=True)
    os.makedirs(folder_grid, exist_ok=True)

    # --- 1. Individual plots ---
    for _, row in df_tau.iterrows():
        fig, ax = plt.subplots()
        filepath = os.path.join(folder_read, row["filename"])
        plot_error_vector_from_file(filepath, ax=ax, linestyle=tau_to_linestyle[tau])
        ax.axhline(0.05, linestyle=":", color="black", linewidth=2.5, alpha=1.0)
        ax.text(0.98, 0.05, "5%", color="black", fontsize=11,
            verticalalignment="bottom", horizontalalignment="right",
            transform=ax.get_yaxis_transform())
        ax.set_title(fr"{tau_label}: $\lambda$={row['lam']:.3f}, $\beta$={row['beta']:.3f}, Auxiliary points per $\lambda$:{row['auxpts']}", fontsize=14)
        ax.set_xlabel("Point Index")
        ax.set_ylabel(r"$e_{\mathrm{max}}$")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(folder_individual, row["filename"].replace(".csv", ".png")), dpi=150)
        plt.close(fig)

    # --- 2. Grid of subplots ---
    n = len(df_tau)
    ncols = min(4, n) if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, (_, row) in enumerate(df_tau.iterrows()):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        filepath = os.path.join(folder_read, row["filename"])
        plot_error_vector_from_file(filepath, ax=ax, linestyle=tau_to_linestyle[tau])
        ax.set_title(fr"$\lambda$={row['lam']:.3f}, $\beta$={row['beta']:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(df_tau), nrows * ncols):
        r, c = divmod(idx, ncols)
        fig.delaxes(axes[r][c])

    fig.suptitle(fr"All raw error vectors for {tau_label}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(folder_grid, f"{tau}_grid.png"), dpi=150)
    plt.close(fig)

# ------------------------------
# Task 2: max_error vs β for each λ
# ------------------------------
summary_folder = os.path.join(folder_out, "summary")
os.makedirs(summary_folder, exist_ok=True)

metric = "max_error"
aux_fixed = int(df["auxpts"].iloc[-1])
ncols = 2
folder_beta = os.path.join(summary_folder, f"{metric}_vs_beta")
os.makedirs(folder_beta, exist_ok=True)

lambdas = sorted(df["lam"].unique())
nrows = (len(lambdas) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

for idx, lam in enumerate(lambdas):
    df_lam = df[(df["lam"] == lam) & (df["auxpts"] == aux_fixed)]
    if df_lam.empty:
        continue

    fig_indiv, ax = plt.subplots()
    for tau in ["tau1", "tau2"]:
        df_sub = df_lam[df_lam["tau_type"] == tau]
        if not df_sub.empty:
            ax.plot(df_sub["beta"], df_sub[metric], label=tau, linestyle=tau_to_linestyle[tau], color=lambda_to_color[lam], marker="o")
    ax.axhline(0.05, linestyle=":", color="red", linewidth=2.0)
    ax.set_title(fr"$e_{{\mathrm{{max}}}}$ vs β (λ = {lam:.3f})")
    ax.set_xlabel("β")
    ax.set_ylabel(r"$e_{\mathrm{max}}$")
    ax.grid(True)
    ax.legend()
    fig_indiv.tight_layout()
    fig_indiv.savefig(os.path.join(folder_beta, f"lam{int(1000*lam):03d}.png"), dpi=150)
    plt.close(fig_indiv)

    r, c = divmod(idx, ncols)
    ax_grid = axes[r][c]
    for tau in ["tau1", "tau2"]:
        df_sub = df_lam[df_lam["tau_type"] == tau]
        if not df_sub.empty:
            ax_grid.plot(df_sub["beta"], df_sub[metric], label=tau, linestyle=tau_to_linestyle[tau], color=lambda_to_color[lam], marker="o")
    ax_grid.set_title(f"λ = {lam:.3f}")
    ax_grid.set_xlabel("β")
    ax_grid.set_ylabel(r"$e_{\mathrm{max}}$")
    ax_grid.grid(True)
    ax_grid.legend()

for i in range(len(lambdas), nrows * ncols):
    r, c = divmod(i, ncols)
    fig.delaxes(axes[r][c])

fig.suptitle(fr"$e_{{\mathrm{{max}}}}$ vs $\beta$ (Auxiliary points per $\lambda$: {aux_fixed})")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(folder_beta, "all_lambdas_grid.png"), dpi=150)
plt.close(fig)

# ------------------------------
# Task 3: max_error vs λ for each fixed β
# ------------------------------
folder_summary = os.path.join(folder_out, "summary", "max_error_vs_lambda")
folder_indiv = os.path.join(folder_summary, "individual")
os.makedirs(folder_indiv, exist_ok=True)

df_fixed_aux = df[df["auxpts"] == aux_fixed]
unique_betas = sorted(df_fixed_aux["beta"].unique())

# --- Individual plots for each β ---
for beta in unique_betas:
    df_beta = df_fixed_aux[df_fixed_aux["beta"] == beta]
    if df_beta.empty:
        continue

    fig, ax = plt.subplots()
    for tau in ["tau1", "tau2"]:
        df_tau = df_beta[df_beta["tau_type"] == tau]
        ax.plot(df_tau["lam"], df_tau["max_error"],
                label=f"{tau}", linestyle=tau_to_linestyle[tau], marker="o")

    ax.set_title(f"Max error vs λ for β = {beta:.3f}")
    ax.set_xlabel("λ")
    ax.set_ylabel(r"$e_{\mathrm{max}}$")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(folder_indiv, f"beta{int(beta*1000):03d}.png"), dpi=150)
    plt.close(fig)

# --- Subplot grid of all β ---
n = len(unique_betas)
ncols = 2
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

for idx, beta in enumerate(unique_betas):
    r, c = divmod(idx, ncols)
    ax = axes[r][c]
    df_beta = df_fixed_aux[df_fixed_aux["beta"] == beta]
    if df_beta.empty:
        fig.delaxes(ax)
        continue

    for tau in ["tau1", "tau2"]:
        df_tau = df_beta[df_beta["tau_type"] == tau]
        ax.plot(df_tau["lam"], df_tau["max_error"],
                label=f"{tau}", linestyle=tau_to_linestyle[tau], marker="o")

    ax.set_title(f"β = {beta:.3f}")
    ax.set_xlabel("λ")
    ax.set_ylabel(r"$e_{\mathrm{max}}$")
    ax.grid(True)
    ax.legend()

for idx in range(n, nrows * ncols):
    r, c = divmod(idx, ncols)
    fig.delaxes(axes[r][c])

fig.suptitle(fr"$e_{{\mathrm{{max}}}}$ vs $\lambda$ for each $\beta$ (Auxiliary points per $\lambda$: {aux_fixed})")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(folder_summary, "all_betas_grid.png"), dpi=150)
plt.close(fig)

# ------------------------------
# Task 4: max_error vs auxpts for each fixed β (Legend on top)
# ------------------------------
folder_summary = os.path.join(folder_out, "summary", "max_error_vs_auxpts_by_beta")
folder_indiv = os.path.join(folder_summary, "individual")
os.makedirs(folder_indiv, exist_ok=True)

unique_betas = sorted(df["beta"].unique())
unique_lambdas = sorted(df["lam"].unique())

# --- Individual plots ---
for beta in unique_betas:
    df_beta = df[df["beta"] == beta]
    if df_beta.empty:
        continue

    fig, ax = plt.subplots()
    for lam in unique_lambdas:
        for tau in ["tau1", "tau2"]:
            df_plot = df_beta[(df_beta["lam"] == lam) & (df_beta["tau_type"] == tau)]
            if not df_plot.empty:
                ax.plot(df_plot["auxpts"], df_plot["max_error"],
                        label=fr"$\lambda = {lam:.3f}$, {tau_labels[tau]}",
                        linestyle=tau_to_linestyle[tau],
                        color=lambda_to_color[lam], marker="o")

    ax.set_title(fr"$e_{{\mathrm{{max}}}}$ vs Auxiliary points per $\lambda$ for $\beta = {beta:.3f}$")
    ax.set_xlabel(r"Auxiliary points per $\lambda$")
    ax.set_ylabel(r"$e_{\mathrm{max}}$")
    ax.grid(True)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(folder_indiv, f"beta{int(beta * 1000):03d}.png"), dpi=150)
    plt.close(fig)

# --- Subplot grid of all β with shared legend just under title ---
ncols = 2
nrows = (len(unique_betas) + ncols - 1) // ncols

fig = plt.figure(figsize=(6 * ncols, 4.8 * nrows + 1.5))  # extra space for legend
gs = GridSpec(nrows=nrows, ncols=ncols, top=0.82, bottom=0.05, hspace=0.3, wspace=0.25)

axes = []
for row in range(nrows):
    for col in range(ncols):
        axes.append(fig.add_subplot(gs[row, col]))

# One legend line per unique (λ, τ)
legend_handles = {}
for idx, beta in enumerate(unique_betas):
    if idx >= len(axes):
        break
    ax = axes[idx]
    df_beta = df[df["beta"] == beta]

    for lam in unique_lambdas:
        for tau in ["tau1", "tau2"]:
            df_plot = df_beta[(df_beta["lam"] == lam) & (df_beta["tau_type"] == tau)]
            if not df_plot.empty:
                label = fr"$\lambda = {lam:.3f}$, {tau_labels[tau]}"
                line, = ax.plot(df_plot["auxpts"], df_plot["max_error"],
                                label=label,
                                linestyle=tau_to_linestyle[tau],
                                color=lambda_to_color[lam], marker="o")
                if label not in legend_handles:
                    legend_handles[label] = line

    ax.axhline(0.05, linestyle=":", color="black", linewidth=2.5, alpha=1.0)
    ax.text(0.98, 0.05, "5%", color="black", fontsize=11,
        verticalalignment="bottom", horizontalalignment="right",
        transform=ax.get_yaxis_transform())
    ax.set_title(fr"$\beta = {beta:.3f}$", fontsize=14)
    ax.set_xlabel(r"Auxiliary points per $\lambda$")
    ax.set_ylabel(r"$e_{\mathrm{max}}$")
    ax.grid(True)

# Delete unused axes
for idx in range(len(unique_betas), len(axes)):
    fig.delaxes(axes[idx])

# --- Title and Legend Placement ---
fig.suptitle(r"$e_{\mathrm{max}}$ vs Auxiliary points per $\lambda$ for each $\beta$", fontsize=18, y=0.96)

fig.legend(list(legend_handles.values()), list(legend_handles.keys()),
           loc="upper center",
           bbox_to_anchor=(0.5, 0.92),  # Just under title
           ncol=4, fontsize=12, frameon=False,
           columnspacing=1.2, handletextpad=0.7)

# Save
fig.savefig(os.path.join(folder_summary, "all_betas_grid.png"), dpi=150, bbox_inches='tight')
plt.close(fig)


# ------------------------------
# Task 5: Heatmap of max error across (λ, β) for all auxpts
# ------------------------------
folder_heatmap = os.path.join(folder_out, "summary", "heatmaps")
os.makedirs(folder_heatmap, exist_ok=True)

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Define LaTeX labels for τ
tau_latex = {"tau1": r"$\tau_1$", "tau2": r"$\tau_2$"}

# Define colormap and normalization
colors = ["steelblue", "whitesmoke", "tomato"]
cmap = LinearSegmentedColormap.from_list("threshold_cmap", colors)

for aux in sorted(df["auxpts"].unique()):
    df_aux = df[df["auxpts"] == aux]

    for tau in ["tau1", "tau2"]:
        df_tau = df_aux[df_aux["tau_type"] == tau]
        heatmap_data = df_tau.pivot(index="lam", columns="beta", values="max_error")
        if heatmap_data.empty or heatmap_data.isnull().all().all():
            print(f"[SKIP] No data for heatmap {tau}, auxpts = {aux}")
            continue

        vmin = heatmap_data.min().min()
        vmax = heatmap_data.max().max()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.05, vmax=vmax)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".2e",
                    cmap=cmap, norm=norm, ax=ax)

        ax.set_title(
            fr"Heatmap of $e_{{\mathrm{{max}}}}$ ({tau_latex[tau]}, Auxiliary points per $\lambda$: ${aux}$)",
        )
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\lambda$")
        fig.tight_layout()
        fname = f"heatmap_max_error_{tau}_auxpts{aux}.png"
        fig.savefig(os.path.join(folder_heatmap, fname), dpi=150)
        plt.close(fig)

