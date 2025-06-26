import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import pandas as pd

def load_params(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_TE_TM_vectors(k_vec, beta):
    k_hat = np.array(k_vec) / np.linalg.norm(k_vec)

    # Reference vector for defining polarization plane
    ref_vec = np.array([0, 0, 1])
    
    # If k is parallel or anti-parallel to z-axis, choose a different ref_vec
    if np.allclose(np.abs(np.dot(k_hat, ref_vec)), 1.0):
        ref_vec = np.array([1, 0, 0])

    TE = np.cross(k_hat, ref_vec)
    TE /= np.linalg.norm(TE)
    
    TM = np.cross(TE, k_hat)
    TM /= np.linalg.norm(TM)

    # Final polarization vector
    pol_vec = np.cos(beta) * TE + np.sin(beta) * TM
    return pol_vec


def material_function(p, bump_dict, substrate_material, is_reference):
    if is_reference:
        return mp.air
    z_surface = sum(
        bump["height"] * np.exp(-((p.x - bump["x0"])**2 + (p.y - bump["y0"])**2) / (2 * bump["sigma"]**2))
        for bump in bump_dict
    )
    return substrate_material if p.z < z_surface else mp.air

def run_meep_sim(params, beta, is_reference=False):
    bump_dict = params["bumpData"]
    halfWidth_x = params["halfWidth_x"]
    halfWidth_y = params["halfWidth_y"]
    halfWidth_z = params["halfWidth_z"]
    resolution = params["resolution"]
    epsilon1 = params["epsilon1"]
    omega = params["omega"]
    k_vector = np.array(params["k"])

    pml_thickness = 2
    cell_x = halfWidth_x + 2 * pml_thickness
    cell_y = halfWidth_y + 2 * pml_thickness
    cell_z = halfWidth_z + 2 * pml_thickness
    cell_size = mp.Vector3(cell_x, cell_y, cell_z)
    frequency = omega
    substrate_material = mp.Medium(epsilon=epsilon1)

    pol_vec = compute_TE_TM_vectors(k_vector, beta)
    Ex_amp, Ey_amp, Ez_amp = pol_vec

    sources = [
        mp.Source(mp.ContinuousSource(frequency=frequency, is_integrated=True),
                  component=mp.Ex, amplitude=Ex_amp,
                  center=mp.Vector3(0, 0, 0.5 * cell_z - pml_thickness - 0.5),
                  size=mp.Vector3(cell_x, cell_y, 0)),
        mp.Source(mp.ContinuousSource(frequency=frequency, is_integrated=True),
                  component=mp.Ey, amplitude=Ey_amp,
                  center=mp.Vector3(0, 0, 0.5 * cell_z - pml_thickness - 0.5),
                  size=mp.Vector3(cell_x, cell_y, 0)),
        mp.Source(mp.ContinuousSource(frequency=frequency, is_integrated=True),
                  component=mp.Ez, amplitude=Ez_amp,
                  center=mp.Vector3(0, 0, 0.5 * cell_z - pml_thickness - 0.5),
                  size=mp.Vector3(cell_x, cell_y, 0)),
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        boundary_layers=[mp.PML(pml_thickness)],
        sources=sources,
        default_material=lambda p: material_function(p, bump_dict, substrate_material, is_reference)
    )

    flux_regions = {
        'top':    mp.FluxRegion(center=mp.Vector3(0, 0, 0.5*cell_z - pml_thickness - 1.0),
                                size=mp.Vector3(cell_x - 2*- pml_thickness, cell_y - 2*pml_thickness, 0)),
        'bottom': mp.FluxRegion(center=mp.Vector3(0, 0, -0.5*cell_z + pml_thickness + 0.1),
                                size=mp.Vector3(cell_x - 2*pml_thickness, cell_y - 2*pml_thickness, 0)),
        'left':   mp.FluxRegion(center=mp.Vector3(-0.5*cell_x + pml_thickness + 0.1, 0, 0),
                                size=mp.Vector3(0, cell_y - 2*pml_thickness, cell_z - 2*pml_thickness)),
        'right':  mp.FluxRegion(center=mp.Vector3(0.5*cell_x - pml_thickness - 0.1, 0, 0),
                                size=mp.Vector3(0, cell_y - 2*pml_thickness, cell_z - 2*pml_thickness)),
        'front':  mp.FluxRegion(center=mp.Vector3(0, -0.5*cell_y + pml_thickness + 0.1, 0),
                                size=mp.Vector3(cell_x - 2*pml_thickness, 0, cell_z - 2*pml_thickness)),
        'back':   mp.FluxRegion(center=mp.Vector3(0, 0.5*cell_y - pml_thickness - 0.1, 0),
                                size=mp.Vector3(cell_x - 2*pml_thickness, 0, cell_z - 2*pml_thickness)),
    }


    flux_monitors = {name: sim.add_flux(frequency, 0, 1, region)
                     for name, region in flux_regions.items()}
    
    sim.run(until=60)

    return {name: mp.get_fluxes(mon)[0] for name, mon in flux_monitors.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("param_path", type=str, help="Path to surfaceParams.json")
    args = parser.parse_args()

    params = load_params(args.param_path)
    betas = params.get("betas")             # <- from the JSON file

    
    scattered_powers = []

    all_ref_flux = {}
    all_full_flux = {}


    for beta in betas:
        print(f"\n📐 Polarization angle β = {np.degrees(beta):.1f}°")
        ref_flux = run_meep_sim(params, beta, is_reference=True)
        full_flux = run_meep_sim(params, beta, is_reference=False)
        all_ref_flux[beta] = ref_flux
        all_full_flux[beta] = full_flux
        scattered_flux = {k: full_flux[k] - ref_flux[k] for k in full_flux}
        total_scattered = sum(scattered_flux.values())
        incident_power = abs(ref_flux["bottom"])
        scattered_powers.append(total_scattered / incident_power)

    plt.figure(dpi=120)
    plt.plot(np.degrees(betas), np.array(scattered_powers) * 100, marker='o')
    plt.xlabel("Polarization angle β (degrees)")
    plt.ylabel("Normalized scattered power (%)")
    plt.title("Scattered power vs polarization")
    plt.grid(True)
    os.makedirs("SimData", exist_ok=True)
    plt.savefig("SimData/scattered_vs_polarization.png")
    plt.show()
    print("✅ Done and saved plot to SimData/scattered_vs_polarization.png")
    
    # Assemble into a DataFrame
    records = []
    for beta in betas:
        for m in all_ref_flux[beta].keys():
            ref_val = all_ref_flux[beta][m]
            full_val = all_full_flux[beta][m]
            diff = full_val - ref_val
            records.append({
                "beta (rad)": beta,
                "monitor": m,
                "ref_flux": ref_val,
                "full_flux": full_val,
                "scattered_flux": diff
            })

    df = pd.DataFrame(records)
    df.to_csv("SimData/flux_comparison.csv", index=False)
    print("✅ Flux comparison saved to SimData/flux_comparison.csv")


if __name__ == "__main__":
    main()
