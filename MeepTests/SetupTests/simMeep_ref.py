import meep as mp
import numpy as np
import json

def run_meep_sim(is_reference=False):
    # --- Load surface parameters ---
    with open("SurfaceData/surfaceParams.json", "r") as f:
        surfaceParams = json.load(f)

    bump_dict = surfaceParams["bumpData"]
    width = surfaceParams["width"]
    resolution = surfaceParams["resolution"]
    epsilon1 = surfaceParams["epsilon1"]
    omega = surfaceParams["omega"]
    k_vector = surfaceParams["k"]
    pml_thickness = 2
    dim = width + 2 * pml_thickness

    cell_x = cell_y = cell_z = dim
    frequency = omega
    cell_size = mp.Vector3(cell_x, cell_y, cell_z)
    substrate_material = mp.Medium(epsilon=epsilon1)

    def material_function(p):
        if is_reference:
            return mp.air
        z_surface = 0.0
        for bump in bump_dict:
            x0 = bump["x0"]
            y0 = bump["y0"]
            h = bump["height"]
            sigma = bump["sigma"]
            r2 = (p.x - x0)**2 + (p.y - y0)**2
            z_surface += h * np.exp(-r2 / (2 * sigma**2))
        return substrate_material if p.z < z_surface else mp.air

    # --- Source definition ---
    sources = [
        mp.EigenModeSource(
            src=mp.ContinuousSource(frequency=frequency, is_integrated=True),
            center=mp.Vector3(0, 0, 0.5 * cell_z - pml_thickness - 0.5),
            size=mp.Vector3(cell_x, cell_y, 0),
            direction=mp.Z,
            eig_kpoint=mp.Vector3(*k_vector),
            eig_band=1,
            eig_match_freq=True
        )
    ]

    # --- PML boundaries ---
    pml_layers = [mp.PML(thickness=pml_thickness, direction=d) for d in [mp.X, mp.Y, mp.Z]]

    # --- Create simulation ---
    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        boundary_layers=pml_layers,
        sources=sources,
        default_material=material_function
    )

    # --- Flux monitors ---
    flux_regions = {
        'top':    mp.FluxRegion(center=mp.Vector3(0, 0, 0.5*cell_z - pml_thickness - 0.1), size=mp.Vector3(cell_x, cell_y, 0)),
        'bottom': mp.FluxRegion(center=mp.Vector3(0, 0, -0.5*cell_z + pml_thickness + 0.1), size=mp.Vector3(cell_x, cell_y, 0)),
        'left':   mp.FluxRegion(center=mp.Vector3(-0.5*cell_x + pml_thickness + 0.1, 0, 0), size=mp.Vector3(0, cell_y, cell_z)),
        'right':  mp.FluxRegion(center=mp.Vector3(0.5*cell_x - pml_thickness - 0.1, 0, 0), size=mp.Vector3(0, cell_y, cell_z)),
        'front':  mp.FluxRegion(center=mp.Vector3(0, -0.5*cell_y + pml_thickness + 0.1, 0), size=mp.Vector3(cell_x, 0, cell_z)),
        'back':   mp.FluxRegion(center=mp.Vector3(0, 0.5*cell_y - pml_thickness - 0.1, 0), size=mp.Vector3(cell_x, 0, cell_z)),
    }

    flux_monitors = {name: sim.add_flux(frequency, 0, 1, region)
                     for name, region in flux_regions.items()}

    sim.run(until=60)

    flux_vals = {name: mp.get_fluxes(mon)[0] for name, mon in flux_monitors.items()}
    return flux_vals


# --- Run reference simulation ---
print("📡 Running reference simulation...")
ref_flux = run_meep_sim(is_reference=True)

# --- Run full surface simulation ---
print("🧱 Running full surface simulation...")
full_flux = run_meep_sim(is_reference=False)

# --- Compute scattered power ---
scattered_flux = {face: full_flux[face] - ref_flux[face] for face in ref_flux.keys()}
total_scattered_power = sum(scattered_flux.values())
incident_power = ref_flux["bottom"]

# --- Normalize ---
scattered_fraction = total_scattered_power / incident_power

# --- Print results ---
print("\n--- Flux Comparison ---")
for face in ref_flux:
    print(f"{face.capitalize():<8} | Ref: {ref_flux[face]:10.4f} | Full: {full_flux[face]:10.4f} | Scattered: {scattered_flux[face]:+10.4f}")

print(f"\n🔁 Total scattered power: {total_scattered_power:.4f}")
print(f"📏 Incident power (reference bottom): {incident_power:.4f}")
print(f"✅ Normalized scattered power: {scattered_fraction:.4%}")
