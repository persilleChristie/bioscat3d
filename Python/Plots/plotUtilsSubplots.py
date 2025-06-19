import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_png_grid(folder_path, n=None, m=None, title=None, figsize=(15, 10), cmap=None):
    """
    Plot all .png images in a folder as n x m subplots.
    If n and m are not given, they are determined automatically.

    Parameters:
    - folder_path (str): Path to the folder containing PNG files.
    - n (int, optional): Number of rows.
    - m (int, optional): Number of columns.
    - title (str, optional): Title for the entire figure.
    - figsize (tuple): Size of the entire figure.
    - cmap (str or None): Optional colormap (e.g., 'gray').
    """

    png_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    num_files = len(png_files)

    if num_files == 0:
        print("No PNG files found in the folder.")
        return

    # Automatically determine grid shape if not provided
    if n is None or m is None:
        m = math.ceil(math.sqrt(num_files))
        n = math.ceil(num_files / m)

    fig, axes = plt.subplots(n, m, figsize=figsize)
    axes = axes.flatten() if num_files > 1 else [axes]

    for i, file in enumerate(png_files):
        ax = axes[i]
        img_path = os.path.join(folder_path, file)
        img = mpimg.imread(img_path)
        ax.imshow(img, cmap=cmap)
        ax.set_title(os.path.splitext(file)[0], fontsize=9)
        ax.axis('off')

    for j in range(num_files, n * m):
        axes[j].axis('off')

    if title:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plot_png_grid(
    folder_path="/home/tripleskull/bioscat3d/Python/Plots/TangentialErrorsEFixedUnitsFromTenDynamic/summary/max_error_vs_auxpts_by_beta",
    title="Max Error vs Auxpts by Beta"
)
