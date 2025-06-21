import numpy as np
import pandas as pd
import json
import os

def generate_surface_params(
    keyword = "One", # postfix to name JSON file
    halfWidth_x=0.5,#*1e-6, # 1.50
    halfWidth_y=0.5,#*1e-6, # 1.50
    halfWidth_z=2.0,#*1e-6,
    resolution=10, # Meep parameter  
    pml_thickness = 2, # Meep parameter
    seed=42, # For surface bump creation
    monitor_size = 1, # monitors are placed with center at +/- monitor_size/2 in all directions
    num_bumps=1,
    uniform=False, # If True, bumps are uniformly distributed in the area
    heights_bumps = [0.07, 0.8],#*1e-6, #should be in [20, 150] nm
    sigmas_bumps = [0.20, 0.030],#*1e-6, #should correspond to lambda < width bumps, sigme < lambda/4
    epsilon1 = 2.56, # substrate epsilon value ???
    alpha = 0.86, # disctance scale for source points in MAS
    omega = 1.0, # What does this represent
    k = [0 ,0 , -1.0],
    numBetas = 4, # makes unidistant betas on the interval [0, pi/2]
    lambda_n = 7, # number of wavelengths to be used in the simulation
    minLambda = 0.20,#*1e-6, # minimum wavelength 0.25
    maxLambda = 0.80,#*1e-6, # maximum wavelength 0.8
):
    
    filename = "surfaceParams" + keyword + ".json"

    # Normalize k vector
    k = np.array(k, dtype=float)
    k = (k / np.linalg.norm(k)).tolist()  # Convert to plain Python list after normalization


    bumpData = []

    if (uniform):
        noise_level = 0.1  # fraction of grid spacing for noise

        # Determine grid size (closest to square)
        nx = int(np.ceil(np.sqrt(num_bumps)))
        ny = int(np.ceil(num_bumps / nx))

        x_vals = np.linspace(-halfWidth_x, halfWidth_x, nx)
        y_vals = np.linspace(-halfWidth_y, halfWidth_y, ny)
        dx = x_vals[1] - x_vals[0] if nx > 1 else 2 * halfWidth_x
        dy = y_vals[1] - y_vals[0] if ny > 1 else 2 * halfWidth_y

        bumpData = []
        np.random.seed(seed)
        count = 0
        for x in x_vals:
            for y in y_vals:
                if count >= num_bumps:
                    break
                x0 = x + np.random.uniform(-noise_level * dx, noise_level * dx)
                y0 = y + np.random.uniform(-noise_level * dy, noise_level * dy)
                height = np.random.uniform(heights_bumps[0], heights_bumps[1])
                sigma = np.random.uniform(sigmas_bumps[0], sigmas_bumps[1])
                bumpData.append({"x0": x0, "y0": y0, "height": height, "sigma": sigma})
                count += 1
            if count >= num_bumps:
                break
    else:
        for _ in range(num_bumps):
            x0 = np.random.uniform(-halfWidth_x, halfWidth_x)
            y0 = np.random.uniform(-halfWidth_y, halfWidth_y)
            height = np.random.uniform(heights_bumps[0], heights_bumps[1])
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
