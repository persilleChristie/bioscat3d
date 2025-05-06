import meep as mp
import numpy as np
import json
import os
import matplotlib.pyplot as plt


def load_params(path="SurfaceData/surfaceParams.json"):
    with open(path, "r") as f:
        return json.load(f)

params = load_params()

# Cell and resolution
pml_thickness = params["pml_thickness"]
half_x = params["halfWidth_x"]
half_y = params["halfWidth_y"]
half_z = params["halfWidth_z"]
cell_size = mp.Vector3(half_x + 2*pml_thickness,
                       half_y + 2*pml_thickness,
                       half_z + 2*pml_thickness)
resolution = params["resolution"]

# Material
epsilon = params["epsilon1"]
substrate = mp.Medium(epsilon=epsilon)
bumps = params["bumpData"]

def material_function(p):
    return mp.air

# PML
boundary_layers = [mp.PML(pml_thickness)]

print("✅ Step 1 complete: domain, resolution, material function, and PML set up.")

# Frequency setup
omega = params["omega"]
k_vec = np.array(params["k"])
k_hat = k_vec / np.linalg.norm(k_vec)

# Flux monitor setup
monitor_thickness = 0.1
cell_x, cell_y, cell_z = cell_size.x, cell_size.y, cell_size.z
flux_regions = {
    "top":    mp.FluxRegion(center=mp.Vector3(0, 0, 0.5*cell_z - pml_thickness - monitor_thickness),
                             size=mp.Vector3(cell_x - 2*pml_thickness, cell_y - 2*pml_thickness, 0)),
    "bottom": mp.FluxRegion(center=mp.Vector3(0, 0, -0.5*cell_z + pml_thickness + monitor_thickness),
                             size=mp.Vector3(cell_x - 2*pml_thickness, cell_y - 2*pml_thickness, 0)),
    "left":   mp.FluxRegion(center=mp.Vector3(-0.5*cell_x + pml_thickness + monitor_thickness, 0, 0),
                             size=mp.Vector3(0, cell_y - 2*pml_thickness, cell_z - 2*pml_thickness)),
    "right":  mp.FluxRegion(center=mp.Vector3(0.5*cell_x - pml_thickness - monitor_thickness, 0, 0),
                             size=mp.Vector3(0, cell_y - 2*pml_thickness, cell_z - 2*pml_thickness)),
    "front":  mp.FluxRegion(center=mp.Vector3(0, -0.5*cell_y + pml_thickness + monitor_thickness, 0),
                             size=mp.Vector3(cell_x - 2*pml_thickness, 0, cell_z - 2*pml_thickness)),
    "back":   mp.FluxRegion(center=mp.Vector3(0, 0.5*cell_y - pml_thickness - monitor_thickness, 0),
                             size=mp.Vector3(cell_x - 2*pml_thickness, 0, cell_z - 2*pml_thickness)),
}

os.makedirs("SimData", exist_ok=True)

# Storage for flux results
beta_list = []
flux_results = {key: [] for key in flux_regions.keys()}

for beta in params["betas"]:
    print(f"\n▶️ Sweeping beta = {np.degrees(beta):.1f} degrees")

    # Polarization basis (TE/TM)
    ref_vec = np.array([0, 0, 1]) if not np.allclose(k_hat, [0, 0, 1]) else np.array([1, 0, 0])
    TE = np.cross(k_hat, ref_vec)
    TE /= np.linalg.norm(TE)
    TM = np.cross(TE, k_hat)
    TM /= np.linalg.norm(TM)
    pol_vec = np.cos(beta) * TE + np.sin(beta) * TM
    Ex_amp, Ey_amp, Ez_amp = pol_vec

    source_offset = 0.5 * (half_x + half_y + half_z)
    #source_offset = 0.5 * np.dot(np.abs(k_hat), [half_x, half_y, half_z])
    source_center = mp.Vector3(*(-k_hat * source_offset))
    sources = []
    for comp, amp in zip([mp.Ex, mp.Ey, mp.Ez], [Ex_amp, Ey_amp, Ez_amp]):
        if abs(amp) > 1e-6:
            sources.append(mp.Source(
                src=mp.ContinuousSource(frequency=omega, is_integrated=True),
                center=source_center,
                component=comp,
                size=mp.Vector3(),
                amplitude=amp
            ))

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        boundary_layers=boundary_layers,
        sources=sources,
        default_material=material_function
    )

    flux_monitors = {name: sim.add_flux(omega, 0, 1, region) for name, region in flux_regions.items()}

    sim.init_sim()
    dft_fields = sim.add_dft_fields(
        [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz],
        [omega],
        where=mp.Volume(center=mp.Vector3(), size=sim.cell_size)
    )

    sim.run(until=60)

    flux_values = {name: mp.get_fluxes(mon)[0] for name, mon in flux_monitors.items()}
    print("\n--- Flux Monitor Results ---")
    for name, val in flux_values.items():
        print(f"{name.capitalize():<8}: {val: .4f}")
        flux_results[name].append(val)

    beta_list.append(np.degrees(beta))

    # Save frequency-domain fields
    field_data = {}
    for comp in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        mp_comp = getattr(mp, comp)
        array_freq = sim.get_dft_array(dft_fields, mp_comp, 0)
        field_data[comp] = array_freq

    # Save all fields in a single .npz file
    beta_deg = int(np.degrees(beta))
    np.savez(f"SimData/fields_beta{beta_deg}_air.npz", **field_data)
    print(f"💾 Saved combined E & H fields for β = {beta_deg}°")

import csv

# Step 1: Compute max flux for each region
max_fluxes = {
    region: max(values) if max(values) > 0 else 1.0
    for region, values in flux_results.items()
}

# Step 2: Build combined rows of raw + normalized values
csv_rows = []
for i, beta_deg in enumerate(beta_list):
    row = {"beta_deg": beta_deg}
    for region, values in flux_results.items():
        raw = values[i]
        norm = raw / max_fluxes[region]
        row[region] = raw
        row[f"{region}_norm"] = norm
    csv_rows.append(row)

# Step 3: Save to CSV
csv_path = "SimData/flux_summary_air.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"📄 Saved flux summary (raw + max-normalized) to {csv_path}")


# Plot flux vs beta for each monitor
plt.figure(figsize=(12, 10))
for i, (name, values) in enumerate(flux_results.items(), 1):
    plt.subplot(3, 2, i)
    plt.plot(beta_list, values, marker='o')
    plt.title(f"{name.capitalize()} flux vs Beta")
    plt.xlabel("Beta (degrees)")
    plt.ylabel("Flux")
    plt.grid(True)
plt.tight_layout()
plt.savefig("SimData/flux_vs_beta_air.png")
print("📍 Saved flux vs beta plots to SimData/flux_vs_beta_air.png")

plt.figure(figsize=(12, 10))
for i, region in enumerate(flux_results.keys(), 1):
    norm_values = [row[f"{region}_norm"] for row in csv_rows]
    plt.subplot(3, 2, i)
    plt.plot(beta_list, norm_values, marker='o')
    plt.title(f"{region.capitalize()} flux (normalized) vs Beta")
    plt.xlabel("Beta (degrees)")
    plt.ylabel("Normalized Flux")
    plt.grid(True)
plt.tight_layout()
plt.savefig("SimData/flux_vs_beta_normalized_air.png")
print("📍 Saved normalized flux plot to SimData/flux_vs_beta_normalized_air.png")

for row in csv_rows:
    row["total_flux"] = sum(row[region] for region in flux_results.keys())


