import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os

# --------------------------------------------
# 1. Load Error Summary from CSVs
# --------------------------------------------
def load_error_summary_df(folder_read):
    pattern = os.path.join(folder_read, "E_tangential_error1_*.csv")
    regex = r"E_tangential_error1_auxpts(\d+)_lambda(\d+)_beta(\d+)\.csv"
    rows = []
    for file in glob.glob(pattern):
        match = re.search(regex, os.path.basename(file))
        if match:
            auxpts = int(match.group(1))
            lam_val = int(match.group(2)) / 1000
            beta = int(match.group(3)) / 1000
            data = pd.read_csv(file, header=None).squeeze("columns")
            row = {
                "filename": os.path.basename(file),
                "auxpts": auxpts,
                "lam": round(lam_val, 6),
                "beta": beta,
                "max_error": data.max(),
                "mean_error": data.mean(),
                "median_error": data.median(),
                "min_error": data.min(),
                "std_error": data.std(),
                "n_points": len(data),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["lam", "auxpts", "beta"]).reset_index(drop=True)
    return df

# --------------------------------------------
# 2. Aggregate Error Metrics Across β
# --------------------------------------------
def aggregate_summary_over_beta(df):
    df = df.copy()
    df["percentile_99"] = df["mean_error"] + 2.326 * df["std_error"]
    df["lam"] = df["lam"].round(6)
    return df.groupby(["lam", "auxpts"], as_index=False).agg({
        "max_error": "max",
        "percentile_99": "max"
    })

# --------------------------------------------
# 3. Plotting: Error vs. AuxPts for each λ
# --------------------------------------------
def plot_max_error_vs_auxpts(df, output_path=None, title=None, label_prefix="λ", hline=0.05):
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for lam_val, subdf in df.groupby("lam"):
        subdf_sorted = subdf.sort_values("auxpts")
        plt.plot(subdf_sorted["auxpts"], subdf_sorted["max_error"],
                 marker="o", label=f"{label_prefix} = {lam_val:.3f}")
    if hline is not None:
        plt.axhline(hline, linestyle="--", color="red", linewidth=2)
    plt.title(title or "Error vs. Auxiliary Points")
    plt.xlabel("Number of auxiliary points")
    plt.ylabel("Error magnitude")
    plt.grid(True)
    plt.legend(title=label_prefix)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

# --------------------------------------------
# 4a. Individual Error Vectors
# --------------------------------------------
def plot_all_error_vectors(df, folder_read, folder_write):
    os.makedirs(folder_write, exist_ok=True)
    for _, row in df.iterrows():
        filepath = os.path.join(folder_read, row["filename"])
        data = pd.read_csv(filepath, header=None).squeeze("columns")
        plt.figure(figsize=(8, 4))
        plt.plot(data)
        plt.axhline(0.05, linestyle="--", color="red", linewidth=1.1)
        plt.title(f"$E^1_{{tangential}}$: auxpts={row['auxpts']}, λ={row['lam']:.3f}, β={row['beta']:.3f}")
        plt.xlabel("Point index")
        plt.ylabel("Error magnitude")
        plt.grid(True)
        plt.tight_layout()
        outname = os.path.join(folder_write, f"Et1_auxpts{row['auxpts']}_lambda{row['lam']:.3f}_beta{row['beta']:.3f}.png")
        plt.savefig(outname)
        plt.close()

# --------------------------------------------
# 4b. Grouped Error Vectors as Subplots
# --------------------------------------------
def plot_all_error_vectors_grid(df, folder_read, folder_write, ncols=4):
    os.makedirs(folder_write, exist_ok=True)
    n = len(df)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
    for idx, (_, row) in enumerate(df.iterrows()):
        r, c = divmod(idx, ncols)
        data = pd.read_csv(os.path.join(folder_read, row["filename"]), header=None).squeeze("columns")
        ax = axes[r][c]
        ax.plot(data)
        ax.axhline(0.05, linestyle="--", color="red", linewidth=1.1)
        ax.set_title(f"auxpts={row['auxpts']}, λ={row['lam']:.3f}, β={row['beta']:.3f}", fontsize=8)
        ax.set_xlabel("Idx")
        ax.set_ylabel("Err")
        ax.grid(True)
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        fig.delaxes(axes[r][c])
    fig.suptitle("All τ₁ Error Vectors", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(folder_write, "tau1_all_vectors_grid.png"))
    plt.close()

# --------------------------------------------
# 5. Main Execution
# --------------------------------------------
if __name__ == "__main__":
    folder_fixed = "../../CSV/TangentialErrorsECurvNoGuard/"
    output_root = "../../Python/Plots/TangentialErrorsECurvNoGuard/"
    df_fixed = load_error_summary_df(folder_fixed)
    df_fixed.to_csv(os.path.join(output_root, "auxStats_summary_fixed.csv"), index=False)

    # Plot 1: Aggregated over β
    df_summary = aggregate_summary_over_beta(df_fixed)
    for metric in ["max_error", "percentile_99"]:
        df_plot = df_summary[["lam", "auxpts", metric]].rename(columns={metric: "max_error"})
        plot_max_error_vs_auxpts(
            df_plot,
            output_path=os.path.join(output_root, f"maxErrorVsAuxpts/{metric}_vs_auxpts_fixed_maxOverBeta.png"),
            title=f"{metric.replace('_', ' ').title()} vs. AuxPts (max over β)"
        )

    # Plot 2: For each β
    for beta_val in sorted(df_fixed["beta"].unique()):
        df_beta = df_fixed[df_fixed["beta"] == beta_val]
        beta_str = f"{int(beta_val * 1000):03d}"
        plot_max_error_vs_auxpts(
            df_beta,
            output_path=os.path.join(output_root, f"errorVsAuxptsByBeta/beta{beta_str}.png"),
            title=f"Max Error vs. AuxPts (β = {beta_val:.3f})"
        )

    # Plot 3a: All error vectors individually
    plot_all_error_vectors(df_fixed, folder_read=folder_fixed,
                           folder_write=os.path.join(output_root, "individualErrorsFixed"))

    # Plot 3b: All error vectors as one grouped grid
    plot_all_error_vectors_grid(df_fixed, folder_read=folder_fixed,
                                folder_write=os.path.join(output_root, "errorGridOverview"))
