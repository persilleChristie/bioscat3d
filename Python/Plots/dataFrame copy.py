import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os

def plot_tau_error_triplets(folder_read, folder_write):
    """
    For each auxpts, plot tau1, tau2, and sum of errors in 3 side-by-side subplots.
    """
    pattern1 = os.path.join(folder_read, "E_tangential_error1_*.csv")
    pattern2 = os.path.join(folder_read, "E_tangential_error2_*.csv")
    regex = r"E_tangential_error1_auxpts(\d+)_lambda(\d+)_beta(\d+)\.csv"

    files1 = glob.glob(pattern1)
    files2 = glob.glob(pattern2)

    paired = []

    for file1 in files1:
        match = re.search(regex, os.path.basename(file1))
        if not match:
            continue
        auxpts = int(match.group(1))
        lam_val = int(match.group(2)) / 1000
        beta = int(match.group(3)) / 1000

        basename2 = os.path.basename(file1).replace("error1", "error2")
        file2 = os.path.join(folder_read, basename2)

        if os.path.exists(file2):
            paired.append((file1, file2, auxpts, lam_val, beta))

    os.makedirs(folder_write, exist_ok=True)

    for file1, file2, auxpts, lam, beta in paired:
        err1 = pd.read_csv(file1, header=None).squeeze("columns")
        err2 = pd.read_csv(file2, header=None).squeeze("columns")
        err_sum = err1 + err2

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

        for ax, data, label in zip(axes, [err1, err2, err_sum], [r"$\tau_1$", r"$\tau_2$", r"$\tau_1 + \tau_2$"]):
            ax.plot(data)
            ax.set_title(f"{label}")
            ax.set_xlabel("Point index")
            ax.grid(True)
            ax.axhline(0.05, color='red', linestyle='--', linewidth=2)

        axes[0].set_ylabel("Error magnitude")
        fig.suptitle(f"Errors for auxpts={auxpts}, λ={lam:.3f}, β={beta:.3f}")
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        outname = os.path.join(folder_write, f"tau_triplet_auxpts{auxpts}_lambda{lam:.3f}_beta{beta:.3f}.png")
        plt.savefig(outname)
        plt.close()


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

def plot_error_vs_beta_by_lambda(df, output_dir):
    """
    For each λ, plot max_error, mean_error, median_error, and 99th percentile vs β,
    with one line per auxpts.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()
    df["percentile_99"] = df["mean_error"] + 2.326 * df["std_error"]

    metrics = {
        "max_error": "Max Error",
        "mean_error": "Mean Error",
        "median_error": "Median Error",
        "percentile_99": "99th Percentile Error (approx)"
    }

    for metric, label in metrics.items():
        for lam_val, subdf in df.groupby("lam"):
            plt.figure(figsize=(10, 6))
            for auxpts, df_aux in subdf.groupby("auxpts"):
                df_aux_sorted = df_aux.sort_values("beta")
                plt.plot(df_aux_sorted["beta"], df_aux_sorted[metric],
                         marker="o", label=f"auxpts = {auxpts}")

            plt.axhline(0.05, linestyle="--", color="red", linewidth=2)
            plt.title(f"{label} vs. Polarisation β (λ = {lam_val:.3f})")
            plt.xlabel("Polarisation angle β")
            plt.ylabel(label)
            plt.grid(True)
            plt.legend(title="Auxiliary points")
            plt.tight_layout()

            filename = f"{metric}_vs_beta_lambda{lam_val:.3f}.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

def plot_errors_vs_beta_grouped_by_lambda(df, fixed_auxpts, output_path=None):
    """
    For a fixed number of auxpts, plot error metrics vs. beta for all lambda values
    in shared subplots (max, mean, median, 99th percentile).
    """
    df = df[df["auxpts"] == fixed_auxpts].copy()
    df["percentile_99"] = df["mean_error"] + 2.326 * df["std_error"]

    fig, axs = plt.subplots(1, 4, figsize=(18, 5), sharex=True)
    metrics = {
        "max_error": "Max Error",
        "mean_error": "Mean Error",
        "median_error": "Median Error",
        "percentile_99": "99th Percentile (approx)"
    }

    for lam, group in df.groupby("lam"):
        for ax, (metric, title) in zip(axs, metrics.items()):
            ax.plot(group["beta"], group[metric], marker="o", label=f"λ = {lam:.3f}")
            ax.axhline(0.05, color="red", linestyle="--", linewidth=2)

    for ax, (_, title) in zip(axs, metrics.items()):
        ax.set_title(title)
        ax.set_xlabel("β")
        ax.set_ylabel("Error")
        ax.grid(True)

    axs[0].legend(title="λ")
    fig.suptitle(f"Tangential Error vs. β (Fixed auxpts = {fixed_auxpts})", fontsize=14)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

# --------------------------------------------
# 2. Plotting: Error vs. AuxPts for each λ
# --------------------------------------------
def plot_max_error_vs_auxpts(df, output_path=None, title=None, label_prefix="λ", hline=0.05):
    """
    Plot max_error vs. auxpts for each λ. Adds red bold dashed line at y=hline.
    """
    plt.figure(figsize=(10, 6))

    if df[["lam", "auxpts"]].duplicated().any():
        print("⚠️ Warning: duplicated (λ, auxpts) pairs in input to plotting function")

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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

# --------------------------------------------
# 3. Plotting: Raw Error Vector Plots
# --------------------------------------------
def plot_all_error_vectors(df, folder_read, folder_write):
    os.makedirs(folder_write, exist_ok=True)

    for _, row in df.iterrows():
        filename = row["filename"]
        auxpts = row["auxpts"]
        lam = row["lam"]
        beta = row["beta"]

        filepath = os.path.join(folder_read, filename)
        data = pd.read_csv(filepath, header=None).squeeze("columns")

        plt.figure(figsize=(8, 4))
        plt.plot(data)
        plt.axhline(0.05, linestyle="--", color="red", linewidth=1.1)

        plt.title(f"$E^1_{{tangential}}$: auxpts={auxpts}, λ={lam:.3f}, β={beta:.3f}")
        plt.xlabel("Point index")
        plt.ylabel("Error magnitude")
        plt.grid(True)
        plt.tight_layout()

        outname = os.path.join(
            folder_write,
            f"Et1_auxpts{auxpts}_lambda{lam:.3f}_beta{beta:.3f}.png"
        )
        plt.savefig(outname)
        plt.close()

def plot_error_vs_beta_fixed_auxpts(df, output_dir, fixed_auxpts):
    """
    For each λ, plot error metrics (max, mean, median, 99%) vs β
    for a fixed number of auxiliary points.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()
    df["percentile_99"] = df["mean_error"] + 2.326 * df["std_error"]
    df = df[df["auxpts"] == fixed_auxpts]

    metrics = {
        "max_error": "Max Error",
        "mean_error": "Mean Error",
        "median_error": "Median Error",
        "percentile_99": "99th Percentile Error (approx)"
    }

    for lam_val, subdf in df.groupby("lam"):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)

        for ax, (metric, label) in zip(axes, metrics.items()):
            sub_sorted = subdf.sort_values("beta")
            ax.plot(sub_sorted["beta"], sub_sorted[metric], marker="o")
            ax.axhline(0.05, linestyle="--", color="red", linewidth=2)
            ax.set_title(label)
            ax.set_xlabel("β")
            ax.set_ylabel("Error")
            ax.grid(True)

        fig.suptitle(f"Error Metrics vs. β for λ = {lam_val:.3f}, auxpts = {fixed_auxpts}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        filename = f"errors_vs_beta_lambda{lam_val:.3f}_auxpts{fixed_auxpts}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()       

# --------------------------------------------
# 4. Aggregation Utilities
# --------------------------------------------
def aggregate_summary_over_beta(df):
    df = df.copy()
    df["percentile_99"] = df["mean_error"] + 2.326 * df["std_error"]
    df["lam"] = df["lam"].round(6)

    summary = (
        df.groupby(["lam", "auxpts"], as_index=False)
          .agg({
              "max_error": "max",
              "mean_error": "max",
              "median_error": "max",
              "percentile_99": "max"
          })
    )
    return summary

def filter_by_beta(df, beta_val):
    return df[df["beta"] == beta_val]

# --------------------------------------------
# 5. Run Script
# --------------------------------------------
if __name__ == "__main__":
    folder_fixed = "../../CSV/TangentialErrorsEFixedRadius/"
    output_root = "../../Python/Plots/auxStats/"
    os.makedirs(output_root, exist_ok=True)

    # Load raw stats
    df_fixed = load_error_summary_df(folder_fixed)
    df_fixed.to_csv(os.path.join(output_root, "auxStats_summary_fixed.csv"), index=False)

    # Aggregate over β
    df_summary = aggregate_summary_over_beta(df_fixed)

    # Plot aggregated metrics with red threshold line
    metrics = {
        "max_error": "Max Error",
        "mean_error": "Mean Error",
        "median_error": "Median Error",
        "percentile_99": "99th Percentile Error (approx)"
    }

    for col, label in metrics.items():
        df_plot = df_summary[["lam", "auxpts", col]].copy()
        df_plot = df_plot.rename(columns={col: "max_error"})
        plot_max_error_vs_auxpts(
            df_plot,
            output_path=os.path.join(output_root, f"Et1_{col}_vs_auxpts_fixed_maxOverBeta.png"),
            title=f"{label} vs. Aux Points (max over β, Fixed Radius)",
            hline=0.05
        )

    # Plot each β separately
    for beta_val in sorted(df_fixed["beta"].unique()):
        df_beta = filter_by_beta(df_fixed, beta_val)
        beta_str = f"{int(beta_val * 1000):03d}"
        plot_max_error_vs_auxpts(
            df_beta,
            output_path=os.path.join(output_root, f"Et1_max_error_vs_auxpts_fixed_beta{beta_str}.png"),
            title=f"Max Tangential Error vs. Aux Points (β = {beta_val:.3f}, Fixed Radius)",
            hline=0.05
        )

    # Plot each raw error vector
    plot_all_error_vectors(
        df_fixed,
        folder_read=folder_fixed,
        folder_write=os.path.join(output_root, "individualErrorsFixed")
    )

    # Plot tau1, tau2, and sum in triplet subplots
    plot_tau_error_triplets(
        folder_read=folder_fixed,
        folder_write=os.path.join(output_root, "tauTripletErrors")
    )

    plot_error_vs_beta_by_lambda(
    df_fixed,
    output_dir=os.path.join(output_root, "vsBeta")
    )

    # === For each fixed auxpts, plot all λ together in shared subplot
    for aux in sorted(df_fixed["auxpts"].unique()):
        plot_errors_vs_beta_grouped_by_lambda(
            df_fixed,
            fixed_auxpts=aux,
            output_path=os.path.join(output_root, f"vsBeta_fixedAuxpts/allLambdas_auxpts{aux}.png")
        )




