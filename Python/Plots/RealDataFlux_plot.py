# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import re
# from glob import glob

# # === Configuration ===
# folder = "../../CSV/FluxRealData/"
# pattern = os.path.join(folder, "ArrayEq_surface*_nrPatches_*_lambdamin_*_lambdamax_*_betanr_*.csv")

# # === Find all matching CSV files ===
# files = sorted(glob(pattern))
# if not files:
#     raise FileNotFoundError("No matching CSV files found in folder.")

# # === Process each file ===
# for filepath in files:
#     filename = os.path.basename(filepath)

#     # Extract metadata from filename
#     match = re.search(r"surface(\d+)_nrPatches_(\d+)_lambdamin_(\d+)_lambdamax_(\d+)_betanr_(\d+)", filename)
#     if not match:
#         print(f"Skipping {filename} — does not match pattern.")
#         continue

#     surface_nr = match.group(1)
#     nrPatches = match.group(2)
#     lambda_min = float(match.group(3)) * 10
#     lambda_max = float(match.group(4)) * 10
#     beta_nr = int(match.group(5))

#     # Read data
#     data = pd.read_csv(filepath, header=None)
#     N, M = data.shape

#     if beta_nr > 2:
#         # Define wavelength and polarization axes
#         # Define axes: wavelengths (vertical) and polarization angles (horizontal)
#         wavelengths = np.linspace(lambda_min, lambda_max, N)
#         polarization_angles = np.linspace(0, np.pi/2, M)  # Adjust if needed

#         # Create meshgrid (transpose mesh to match orientation)
#         A, W = np.meshgrid(polarization_angles, wavelengths)

#         # Create the plot
#         plt.figure(figsize=(6, 5))
#         im = plt.pcolormesh(polarization_angles, wavelengths, data, shading='auto', cmap='jet')
#         cbar = plt.colorbar(im)
#         cbar.set_label('Flux')

#         plt.xlabel("Polarization angle (rad)")
#         plt.ylabel("Wavelength (nm)")
#         plt.title(f"Surface {surface_nr} — Flux vs λ and β")

#         # Save the plot
#         save_path = f"RealData/flux_colormap_surface{surface_nr}_nrPatches{nrPatches}.png"
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=300)
#         plt.close()
#         print(f"Saved {save_path}")


#     else:
#         lambdas = np.linspace(lambda_min, lambda_max, N)

#         # === Create subplots ===
#         fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#         # Plot Fluxes
#         ax_flux = axes[0]
#         ax_flux.plot(lambdas, data[0], label=r'$\beta$ = 0.0')
#         ax_flux.plot(lambdas, data[1], label=r'$\beta$ = $\pi$/2')
#         ax_flux.set_title(f"Fluxes")
#         ax_flux.set_xlabel("Wavelength λ (nm)")
#         ax_flux.set_ylabel("Flux")
#         ax_flux.grid(True)
#         ax_flux.legend()
#         fig.suptitle(f"Surface {surface_nr}, {nrPatches} patches")

#         # Plot Ratio
#         ax_ratio = axes[1]
#         ratio = data[0] / data[1]
#         ax_ratio.plot(lambdas, ratio, label='F1 / F2', color='darkblue')
#         ax_ratio.axhline(1.0, ls='--', color='red', lw=1)
#         ax_ratio.set_title(f"Flux Ratio")
#         ax_ratio.set_xlabel("Wavelength λ (nm)")
#         ax_ratio.set_ylabel("Ratio")
#         ax_ratio.grid(True)

#         plt.tight_layout()

#         # === Save figure ===
#         save_path = f"RealData/flux_surface{surface_nr}_nrPatches{nrPatches}.png"
#         plt.savefig(save_path, dpi=300)
#         print(f"Saved {save_path}")
#         plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from glob import glob

# === Configuration ===
folder = "../../CSV/FluxRealData/"
pattern = os.path.join(folder, "ArrayEq_surface*_*.csv")

# === Find all matching CSV files ===
files = sorted(glob(pattern))
if not files:
    raise FileNotFoundError("No matching CSV files found in folder.")

# === Process each file ===
for filepath in files:
    filename = os.path.basename(filepath)

    # Updated regex pattern to match new filename structure
    match = re.search(
        r"surface(\d+)(variablePatchSize|constantPatchSize_\d+)_lambdamin_(\d+)_lambdamax_(\d+)_betanr_(\d+)",
        filename
    )
    if not match:
        print(f"Skipping {filename} — does not match new pattern.")
        continue

    surface_nr = match.group(1)
    patch_info = match.group(2)
    lambda_min = float(match.group(3)) * 10
    lambda_max = float(match.group(4)) * 10
    beta_nr = int(match.group(5))

    # Make patch_info human-readable for titles
    if patch_info == "variablePatchSize":
        patch_label = "Variable Patch Size"
    else:
        match_patch = re.match(r"constantPatchSize_(\d+)", patch_info)
        if match_patch:
            patch_label = f"Constant Patch Size = {match_patch.group(1)}"
        else:
            patch_label = patch_info  # fallback


    # Read data
    data = pd.read_csv(filepath, header=None)
    N, M = data.shape

    if beta_nr > 2:
        wavelengths = np.linspace(lambda_min, lambda_max, N)
        polarization_angles = np.linspace(0, np.pi/2, M)

        A, W = np.meshgrid(polarization_angles, wavelengths)

        plt.figure(figsize=(6, 5))
        im = plt.pcolormesh(polarization_angles, wavelengths, data, shading='auto', cmap='jet')
        cbar = plt.colorbar(im)
        cbar.set_label('Flux')

        plt.xlabel("Polarization angle (rad)")
        plt.ylabel("Wavelength (nm)")
        plt.title(f"Surface {surface_nr}, {patch_label} — Flux vs λ and β")

        save_path = f"RealData/flux_colormap_surface{surface_nr}_{patch_info}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

    else:
        lambdas = np.linspace(lambda_min, lambda_max, N)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax_flux = axes[0]
        ax_flux.plot(lambdas, data[0], label=r'$\beta$ = 0.0')
        ax_flux.plot(lambdas, data[1], label=r'$\beta$ = $\pi$/2')
        ax_flux.set_title(f"Fluxes")
        ax_flux.set_xlabel("Wavelength λ (nm)")
        ax_flux.set_ylabel("Flux")
        ax_flux.grid(True)
        ax_flux.legend()
        fig.suptitle(f"Surface {surface_nr}, {patch_label}")

        ax_ratio = axes[1]
        ratio = data[0] / data[1]
        ax_ratio.plot(lambdas, ratio, label='F1 / F2', color='darkblue')
        ax_ratio.axhline(1.0, ls='--', color='red', lw=1)
        ax_ratio.set_title(f"Flux Ratio")
        ax_ratio.set_xlabel("Wavelength λ (nm)")
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.grid(True)

        plt.tight_layout()

        save_path = f"RealData/flux_surface{surface_nr}_{patch_info}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Saved {save_path}")
        plt.close()
