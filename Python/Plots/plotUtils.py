import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

def load_error_summary_df(folder_read):
    import os, glob, re
    import pandas as pd

    all_rows = []

    for tau_type in ["1", "2"]:
        pattern = os.path.join(folder_read, f"E_tangential_error{tau_type}_*.csv")
        regex = rf"E_tangential_error{tau_type}_auxpts(\d+)_lambda(\d+)_beta(\d+)\.csv"

        for file in glob.glob(pattern):
            match = re.search(regex, os.path.basename(file))
            if match:
                auxpts = int(match.group(1))
                lam_val = int(match.group(2)) / 1000
                beta = int(match.group(3)) / 1000
                data = pd.read_csv(file, header=None).squeeze("columns")
                row = {
                    "filename": os.path.basename(file),
                    "tau_type": f"tau{tau_type}",
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
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print(f"⚠️ No error CSVs found in {folder_read}")
    return df.sort_values(by=["tau_type", "lam", "auxpts", "beta"]).reset_index(drop=True)

def plot_stat_vs_variable(df, x, y="mean_error", hue="tau_type",
                          ax=None, title=None, linestyle="-", save_path=None, dpi=150):
    """
    Plot an error statistic vs a variable, grouped by a hue (e.g., tau_type).

    Parameters:
        df (pd.DataFrame): DataFrame with summary statistics.
        x (str): Column to use as x-axis (e.g., 'beta', 'lam', 'auxpts').
        y (str): Column to use as y-axis (e.g., 'mean_error', 'max_error').
        hue (str): Column to group by as separate curves (default: 'tau_type').
        ax (matplotlib.axes.Axes, optional): Axis to plot into.
        title (str, optional): Title to use; auto-generated if None.
        linestyle (str): Line style for curves (default: solid '-').
        save_path (str, optional): If given, saves figure to this path.
        dpi (int): Resolution of saved figure if `save_path` is provided.

    Returns:
        matplotlib.axes.Axes: The axis object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Group by hue or use full data
    if hue and hue in df.columns:
        groups = df.groupby(hue)
    else:
        groups = [(None, df)]

    for name, group in groups:
        label = str(name) if name is not None else None
        sorted_group = group.sort_values(by=x)
        ax.plot(sorted_group[x], sorted_group[y], marker="o", linestyle=linestyle, label=label)

    ax.set_xlabel(x)
    ax.set_ylabel(y.replace("_", " ").capitalize())
    ax.set_title(title or f"{y.replace('_', ' ').capitalize()} vs {x}")
    ax.grid(True)

    if hue:
        ax.legend(title=hue)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    else:
        plt.tight_layout()

    return ax

def plot_error_vector_from_file(filepath, ax=None, hline=0.05, title=None, label=None):
    import pandas as pd
    import os
    import re
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    data = pd.read_csv(filepath, header=None).squeeze("columns")
    ax.plot(data.index, data.values, marker=".", linestyle="-", label=label)

    if title is None:
        fname = os.path.basename(filepath)
        match = re.search(r"E_tangential_error(\d)_auxpts(\d+)_lambda(\d+)_beta(\d+)\.csv", fname)
        if match:
            tau = f"τ{match.group(1)}"
            auxpts = int(match.group(2))
            lam = int(match.group(3)) / 1000
            beta = int(match.group(4)) / 1000
            title = f"{tau}: auxpts={auxpts}, λ={lam:.3f}, β={beta:.3f}"
        else:
            title = fname

    ax.set_title(title)
    ax.set_xlabel("Point index")
    ax.set_ylabel("Error magnitude")
    ax.grid(True)

    if hline is not None:
        ax.axhline(hline, linestyle="--", color="red", linewidth=1)

    if label:
        ax.legend()

    return ax

