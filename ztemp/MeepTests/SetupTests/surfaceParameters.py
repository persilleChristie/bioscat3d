import numpy as np
import pandas as pd
import json
import os

def generate_surface_params(
    filename="SurfaceData/surfaceParams.json",
    halfWidth_x=1.0,
    halfWidth_y=1.0,
    halfWidth_z=2.0,
    resolution=10,
    pml_thickness = 2,
    seed=42,
    num_bumps=3,
    epsilon1 = 2.56,
    alpha = 0.86,
    omega=1.0,
    k = [50.0, 50.0, -1.0],
    numBetas = 1 # makes unidistant betas on the interval [0, pi/2]
):
    np.random.seed(seed)

    # Normalize k vector
    k = np.array(k, dtype=float)
    k = (k / np.linalg.norm(k)).tolist()  # Convert to plain Python list after normalization


    bumpData = []
    for _ in range(num_bumps):
        x0 = np.random.uniform(-halfWidth_x, halfWidth_x)
        y0 = np.random.uniform(-halfWidth_y, halfWidth_y)
        height = np.random.uniform(0.1, 0.4)
        sigma = np.random.uniform(0.2, 0.7)
        bumpData.append({"x0": x0, "y0": y0, "height": height, "sigma": sigma})

    betas = np.linspace(0, np.pi/2, numBetas).tolist() 

    surfaceParams = {
        "bumpData": bumpData,
        "seed": seed,
        "halfWidth_x": halfWidth_x,
        "halfWidth_y": halfWidth_y,
        "halfWidth_z": halfWidth_z,
        "resolution": resolution,
        "pml_thickness" : pml_thickness,
        "omega": omega,
        "epsilon1": epsilon1,
        "k": k,
        "betas": betas,
        "alpha": alpha
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(surfaceParams, f, indent=4)

    print(f"✅ Saved surfaceParams to {filename}")

# Example use:
generate_surface_params()
