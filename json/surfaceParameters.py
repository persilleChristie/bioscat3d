import numpy as np
import pandas as pd
import json
import os

def generate_surface_params(
    keyword = "NormalNewGeom", # postfix to name JSON file
    halfWidth_x=1.0, # 1.50
    halfWidth_y=1.0, # 1.50
    halfWidth_z=2.0,
    resolution=10, # Meep parameter  
    pml_thickness = 2, # Meep parameter
    seed=42, # For surface bump creation
    monitor_size = 1, # monitors are placed with center at +/- monitor_size/2 in all directions
    num_bumps=100,
    hights_bumps = [0.02, 0.15], #should be in [20, 150] nm
    sigmas_bumps = [0.02, 0.075], #should correspond to lambde < width bumps, sigme < lambda/4
    epsilon1 = 2.56, # substrate epsilon value ???
    alpha = 0.86, # disctance scale for source points in MAS
    omega = 1.0, # What does this represent
    k = [0 ,0 , -1.0],
    numBetas = 1, # makes unidistant betas on the interval [0, pi/2]
    lambda_n = 1, # number of wavelengths to be used in the simulation
    minLambda = 0.7, # minimum wavelength 0.25
    maxLambda = 0.7, # maximum wavelength 0.8
):
    
    filename = "surfaceParams" + keyword + ".json"

    # Normalize k vector
    k = np.array(k, dtype=float)
    k = (k / np.linalg.norm(k)).tolist()  # Convert to plain Python list after normalization


    bumpData = []
    np.random.seed(seed)
    for _ in range(num_bumps):
        x0 = np.random.uniform(-halfWidth_x, halfWidth_x)
        y0 = np.random.uniform(-halfWidth_y, halfWidth_y)
        height = np.random.uniform(hights_bumps[0], hights_bumps[1])
        sigma = np.random.uniform(sigmas_bumps[0], sigmas_bumps[1])
        bumpData.append({"x0": x0, "y0": y0, "height": height, "sigma": sigma})

    betas = np.linspace(0, np.pi/2, numBetas).tolist() 

    surfaceParams = {
        "bumpData": bumpData,
        "monitor_size": monitor_size, #monitors are placed with center at +/1 monitor_size/2 in all directions
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
        "alpha": alpha,
        "lambda_n": lambda_n,
        "minLambda": minLambda,
        "maxLambda": maxLambda,
    }

    #os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(surfaceParams, f, indent=4)

    print(f"✅ Saved surfaceParams to {filename}")

# Example use:
generate_surface_params()
