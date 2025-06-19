import pandas as pd
import numpy as np

def load_field_csv(filepath, reshape=None):
    """
    Load field data from a CSV file.
    
    Parameters:
        filepath: path to the CSV file
        reshape: optional shape (rows, cols) to reshape flat data into 2D
        
    Returns:
        numpy array (1D or 2D depending on reshape)
    """
    df = pd.read_csv(filepath, header=None)
    data = df.to_numpy().squeeze()

    if reshape:
        return data.reshape(reshape)
    return data

import numpy as np
import matplotlib.pyplot as plt

def plot_near_field(data, title="Near Field |E|", extent=None, save_path=None):
    """
    Plots a 2D near-field intensity map (e.g. |E|).
    Parameters:
        data: 2D numpy array
        title: plot title
        extent: (xmin, xmax, ymin, ymax) for axis scaling
        save_path: if given, save figure to file
    """
    plt.figure(figsize=(6, 5))
    im = plt.imshow(data, origin='lower', cmap='inferno', extent=extent)
    plt.colorbar(im, label='Field magnitude')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_far_field(theta, field, title="Far Field |E(θ)|", save_path=None):
    """
    Plots far-field magnitude in polar coordinates.
    Parameters:
        theta: 1D array of angles in radians
        field: 1D array of magnitudes
        title: plot title
        save_path: if given, save figure to file
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, field)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_flux_bar(flux_dict, title="Flux Through Surfaces", save_path=None):
    """
    Plots a bar chart of flux values per surface.
    Parameters:
        flux_dict: dictionary of surface -> flux_value
        title: plot title
        save_path: if given, save figure to file
    """
    surfaces = list(flux_dict.keys())
    values = list(flux_dict.values())

    plt.figure()
    plt.bar(surfaces, values)
    plt.ylabel("Flux")
    plt.title(title)
    plt.grid(True, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
