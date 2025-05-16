def run_flux_analysis(param_path="SurfaceData/surfaceParams.json", run_reference=False, postfix = ""):
    import meep as mp
    import numpy as np
    import json
    import os
    import matplotlib.pyplot as plt
    import csv

    # Load parameters
    with open(param_path + postfix, "r") as f:
        params = json.load(f)

    # Geometry and resolution
    pml = params["pml_thickness"]
    half_x, half_y, half_z = params["halfWidth_x"], params["halfWidth_y"], params["halfWidth_z"]
    cell_size = mp.Vector3(half_x + 2*pml, half_y + 2*pml, half_z + 2*pml)
    resolution = params["resolution"]

    omega = params["omega"]
    k_vec = np.array(params["k"])
    k_hat = k_vec / np.linalg.norm(k_vec)

    # Define flux monitor regions
    mon_thick = 0.1
    cell_x, cell_y, cell_z = cell_size.x, cell_size.y, cell_size.z
    flux_regions = {
        "top":    mp.FluxRegion(mp.Vector3(0, 0, 0.5*cell_z - pml - mon_thick), size=mp.Vector3(cell_x - 2*pml, cell_y - 2*pml, 0)),
        "bottom": mp.FluxRegion(mp.Vector3(0, 0, -0.5*cell_z + pml + mon_thick), size=mp.Vector3(cell_x - 2*pml, cell_y - 2*pml, 0)),
        "left":   mp.FluxRegion(mp.Vector3(-0.5*cell_x + pml + mon_thick, 0, 0), size=mp.Vector3(0, cell_y - 2*pml, cell_z - 2*pml)),
        "right":  mp.FluxRegion(mp.Vector3(0.5*cell_x - pml - mon_thick, 0, 0), size=mp.Vector3(0, cell_y - 2*pml, cell_z - 2*pml)),
        "front":  mp.FluxRegion(mp.Vector3(0, -0.5*cell_y + pml + mon_thick, 0), size=mp.Vector3(cell_x - 2*pml, 0, cell_z - 2*pml)),
        "back":   mp.FluxRegion(mp.Vector3(0, 0.5*cell_y - pml - mon_thick, 0), size=mp.Vector3(cell_x - 2*pml, 0, cell_z - 2*pml)),
    }

    os.makedirs("SimData", exist_ok=True)

    if run_reference:
        print("📏 Running reference simulation (air only)...")
        ref_flux = {key: [] for key in flux_regions.keys()}
        for beta in params["betas"]:
            # Polarization
            ref_vec = np.array([0, 0, 1]) if not np.allclose(k_hat, [0, 0, 1]) else np.array([1, 0, 0])
            TE = np.cross(k_hat, ref_vec); TE /= np.linalg.norm(TE)
            TM = np.cross(TE, k_hat); TM /= np.linalg.norm(TM)
            pol_vec = np.cos(beta) * TE + np.sin(beta) * TM
            Ex_amp, Ey_amp, Ez_amp = pol_vec

            # Source
            offset = 0.5 * (half_x + half_y + half_z)
            src_center = mp.Vector3(*(-k_hat * offset))
            sources = [mp.Source(mp.ContinuousSource(frequency=omega, is_integrated=True),
                        center=src_center, component=comp, amplitude=amp)
                        for comp, amp in zip([mp.Ex, mp.Ey, mp.Ez], [Ex_amp, Ey_amp, Ez_amp]) if abs(amp) > 1e-6]

            sim = mp.Simulation(cell_size=cell_size, resolution=resolution,
                                boundary_layers=[mp.PML(pml)], sources=sources,
                                default_material=mp.air)

            flux_mons = {name: sim.add_flux(omega, 0, 1, region) for name, region in flux_regions.items()}
            sim.run(until=60)
            for name, mon in flux_mons.items():
                ref_flux[name].append(mp.get_fluxes(mon)[0])

        np.savez("SimData/reference_flux.npz", **ref_flux)
        print("💾 Reference flux saved.")
        return

    # Run main (bumpy surface) simulation
    ref_flux = np.load("SimData/reference_flux.npz")
    bumps = params["bumpData"]
    substrate = mp.Medium(epsilon=params["epsilon1"])
    def material_function(p):
        z = sum(b["height"] * np.exp(-((p.x - b["x0"])**2 + (p.y - b["y0"])**2) / (2 * b["sigma"]**2)) for b in bumps)
        return substrate if p.z < z else mp.air

    print("🌀 Running main simulation (with surface)...")
    beta_list = []
    flux_results = {key: [] for key in flux_regions.keys()}

    for idx, beta in enumerate(params["betas"]):
        print(f"\n▶️ Beta = {np.degrees(beta):.1f}°")

        # Polarization
        ref_vec = np.array([0, 0, 1]) if not np.allclose(k_hat, [0, 0, 1]) else np.array([1, 0, 0])
        TE = np.cross(k_hat, ref_vec); TE /= np.linalg.norm(TE)
        TM = np.cross(TE, k_hat); TM /= np.linalg.norm(TM)
        pol_vec = np.cos(beta) * TE + np.sin(beta) * TM
        Ex_amp, Ey_amp, Ez_amp = pol_vec

        offset = 0.5 * (half_x + half_y + half_z)
        src_center = mp.Vector3(*(-k_hat * offset))
        sources = [mp.Source(mp.ContinuousSource(frequency=omega, is_integrated=True),
                    center=src_center, component=comp, amplitude=amp)
                    for comp, amp in zip([mp.Ex, mp.Ey, mp.Ez], [Ex_amp, Ey_amp, Ez_amp]) if abs(amp) > 1e-6]

        sim = mp.Simulation(cell_size=cell_size, resolution=resolution,
                            boundary_layers=[mp.PML(pml)], sources=sources,
                            default_material=material_function)

        flux_mons = {name: sim.add_flux(omega, 0, 1, region) for name, region in flux_regions.items()}
        sim.init_sim()
        sim.run(until=60)

        # Subtract incident (reference) from total → scattered
        flux_values = {}
        for name, mon in flux_mons.items():
            total = mp.get_fluxes(mon)[0]
            incident = ref_flux[name][idx]
            scattered = total - incident
            flux_values[name] = scattered
            flux_results[name].append(scattered)

        beta_list.append(np.degrees(beta))

        # Save frequency domain fields
        dft_fields = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz], [omega],
                                         where=mp.Volume(center=mp.Vector3(), size=sim.cell_size))
        sim.run(until=1e-3)  # small run to dump fields
        field_data = {comp: sim.get_dft_array(dft_fields, getattr(mp, comp), 0)
                      for comp in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]}
        beta_deg = int(np.degrees(beta))
        np.savez(f"SimData/fields_beta{beta_deg}.npz", **field_data)

    # Save flux CSV
    max_flux = {r: max(abs(vs)) for r, vs in flux_results.items()}
    csv_rows = []
    for i, beta in enumerate(beta_list):
        row = {"beta_deg": beta}
        for r in flux_results:
            raw = flux_results[r][i]
            row[r] = raw
            row[f"{r}_norm"] = raw / max_flux[r]
        row["total_flux"] = sum(row[r] for r in flux_results)
        csv_rows.append(row)

    with open("SimData/flux_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print("📄 Saved scattered flux results to SimData/flux_summary.csv")

    # Plot
    plt.figure(figsize=(12, 10))
    for i, (name, values) in enumerate(flux_results.items(), 1):
        plt.subplot(3, 2, i)
        plt.plot(beta_list, values, marker='o')
        plt.title(f"{name.capitalize()} scattered flux vs Beta")
        plt.xlabel("Beta (degrees)"); plt.ylabel("Flux"); plt.grid(True)
    plt.tight_layout()
    plt.savefig("SimData/flux_vs_beta.png")

    plt.figure(figsize=(12, 10))
    for i, r in enumerate(flux_results, 1):
        norm = [row[f"{r}_norm"] for row in csv_rows]
        plt.subplot(3, 2, i)
        plt.plot(beta_list, norm, marker='o')
        plt.title(f"{r.capitalize()} normalized scattered flux vs Beta")
        plt.xlabel("Beta (degrees)"); plt.ylabel("Normalized"); plt.grid(True)
    plt.tight_layout()
    plt.savefig("SimData/flux_vs_beta_normalized.png")
    print("📊 Plots saved to SimData/")

# Step 1: run reference simulation
run_flux_analysis(run_reference=True)

# Step 2: run actual simulation and compute scattered flux
run_flux_analysis(param_path="SurfaceData/surfaceParams111.json", postfix = "111")

run_flux_analysis(param_path="SurfaceData/surfaceParams11neg1.json", postfix = "11neg1")

run_flux_analysis(param_path="SurfaceData/surfaceParams.json", postfix = "neg1neg1neg1")
