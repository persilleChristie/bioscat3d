import numpy as np
import meep as mp
import pandas as pd
import subprocess
import matplotlib.pyplot as plt



def run_cpp_code(executable_path, param_file, testpoints_file, auxpoint_file, evalpoint_file, E_output_file, H_output_file):
    """
    Runs the C++ executable with the given parameters and testpoints file.
    Assumes the executable takes input/output file paths as command-line arguments.
    """
    try:
        subprocess.run([executable_path, param_file, testpoints_file, auxpoint_file, evalpoint_file, E_output_file, H_output_file], check=True)
        print(f"C++ implementation ran successfully, E field saved to {E_output_file}, H field saved to {H_output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ implementation: {e}")


def plot_difference_abs(file1_E, file2_E, file1_H, file2_H, evalpoints):
    """
    Plots heatmaps of absolute differences in E-fields (and optionally H-fields) between two CSVs.
    
    Parameters:
        file1_E, file2_E: CSV files with E-field vectors (Ex, Ey).
        file1_H, file2_H: CSV files with H-field vectors (Hx, Hy).
        evalpoints: Nx2 array-like of 2D coordinates [x, y] on a structured grid.
    """
    # Convert evalpoints
    evalpoints = np.array(evalpoints)
    x = evalpoints[:, 0]
    y = evalpoints[:, 1]

    # Unique grid
    x_unique = np.unique(x)
    y_unique = np.unique(y)

    if len(x_unique) * len(y_unique) != len(x):
        raise ValueError("evalpoints do not form a complete grid.")

    X, Y = np.meshgrid(x_unique, y_unique)

    def load_and_compute_magnitude(file1, file2):
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        if list(df1.columns) != list(df2.columns):
            raise ValueError(f"CSV column names do not match between {file1} and {file2}")
        diff = np.abs(np.linalg.norm(df1.values, axis=1) - np.linalg.norm(df2.values, axis=1))
        return diff.reshape(len(y_unique), len(x_unique))

    # Compute E-field difference
    Z_E = load_and_compute_magnitude(file1_E, file2_E)

    # Set up subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # E-field plot
    mesh_E = axes[0].pcolormesh(X, Y, Z_E, shading='auto', cmap='inferno')
    fig.colorbar(mesh_E, ax=axes[0], label='Δ|E|')
    axes[0].set_title('E-field Absolute Difference')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].axis('equal')

    # H-field plot 
    Z_H = load_and_compute_magnitude(file1_H, file2_H)
    mesh_H = axes[1].pcolormesh(X, Y, Z_H, shading='auto', cmap='plasma')
    fig.colorbar(mesh_H, ax=axes[1], label='Δ|H|')
    axes[1].set_title('H-field Absolute Difference')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].axis('equal')

    plt.tight_layout()
    plt.show()


if True:
    wavelength = 325e-3    
    polarization = 0
    propagation_vector = [1, 0.5, -1]/np.sqrt(2.25)

    param_data = [["wavelength", wavelength],
                ["polarization",polarization],
                ["incident_x", propagation_vector[0]],
                ["incident_y", propagation_vector[1]],
                ["incident_z", propagation_vector[2]],
                ]  

    evalpoints = []


    # Convert testpoints to DataFrame and add column names
    evalpoints_df = pd.DataFrame(evalpoints, columns=["testpoint_x", "testpoint_y", "testpoint_z"])

    # Convert parameters to DataFrame
    param_df = pd.DataFrame(param_data, columns=["Parameter", "Value"])

    # Save both to CSV
    param_df.to_csv("ForTesting/params.csv", index=False)
    evalpoints_df.to_csv("ForTesting/evalpoints.csv", index=False)


path = "../MeepTests/SurfaceData/"

print("Running C++ implementation...")
run_cpp_code("./FieldTest", "params.csv", path+"test_points.csv", path+"aux_points.csv", "evalpoints.csv", "Ouputs/Ecpp.csv", "Outputs/Hcpp.csv")