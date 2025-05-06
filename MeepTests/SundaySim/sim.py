import meep as mp
import numpy as np
import json
import os
import matplotlib.pyplot as plt

keyword = "Normal"
path = "SurfaceData/surfaceParams" + keyword + ".json"
def load_params(path = path):
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

# PML
boundary_layers = [mp.PML(pml_thickness)]

print("✅ Step 1 complete: domain, resolution, material function, and PML set up.")

# Frequency setup
omega = params["omega"]

# Flux monitor setup
monitor_thickness = 0.1
cell_x, cell_y, cell_z = cell_size.x, cell_size.y, cell_size.z
flux_regions = {
    "top":    mp.FluxRegion(center=mp.Vector3(0, 0, cell_z/8),
                             size=mp.Vector3(cell_x - 2*pml_thickness, cell_y - 2*pml_thickness, 0)),
    "bottom": mp.FluxRegion(center=mp.Vector3(0, 0, -cell_z/8),
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

sources = [mp.Source(
    src=mp.ContinuousSource(frequency=omega, is_integrated=True),
    center=mp.Vector3(0, 0, cell_z/4),
    component=mp.Ex,
    size=mp.Vector3(cell_size.x, cell_size.y, 0),
    #size=mp.Vector3(cell_size.x - 2*pml_thickness, cell_size.y - 2*pml_thickness, 0)
)]




sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    boundary_layers=boundary_layers,
    sources=sources,
    default_material=mp.air
)

sim.init_sim()

flux_monitors = {name: sim.add_flux(omega, 0, 1, region) for name, region in flux_regions.items()}
    
dft_fields = sim.add_dft_fields(
    [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz],
    [omega],
    where=mp.Volume(center=mp.Vector3(), size=sim.cell_size)
)

sim.run(until=60)

# Store flux monitor data for empty simulation
box_pz_data_empty = sim.get_flux_data(flux_monitors["top"])
box_mz_data_empty = sim.get_flux_data(flux_monitors["bottom"])
box_mx_data_empty = sim.get_flux_data(flux_monitors["left"])
box_px_data_empty = sim.get_flux_data(flux_monitors["right"])
box_my_data_empty = sim.get_flux_data(flux_monitors["front"])
box_py_data_empty = sim.get_flux_data(flux_monitors["back"])

flux_values = {name: mp.get_fluxes(mon)[0] for name, mon in flux_monitors.items()}
print("\n--- Flux Monitor Results ---")
for name, val in flux_values.items():
    print(f"{name.capitalize():<8}: {val: .4f}")
    flux_results[name].append(val)

# Save frequency-domain fields
field_data = {}
for comp in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
    mp_comp = getattr(mp, comp)
    array_freq = sim.get_dft_array(dft_fields, mp_comp, 0)
    field_data[comp] = array_freq

# Save all fields in a single .npz file
np.savez(f"SimData/fields.npz", **field_data)
print("💾 Saved combined E & H fields to SimData/fields.npz")

#############################################################

sim.reset_meep()

keyword = "Normal"
path = "SurfaceData/surfaceParams" + keyword + ".json"
def load_params(path = path):
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
epsilon1 = params["epsilon1"] 
substrate = mp.Medium(epsilon = epsilon1)
bumps = params["bumpData"]

def material_function(p):
    z_surface = sum(
        b["height"] * np.exp(-((p.x - b["x0"])**2 + (p.y - b["y0"])**2) / (2 * b["sigma"]**2))
        for b in bumps
    )
    return substrate if p.z < z_surface else mp.air

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
    "top":    mp.FluxRegion(center=mp.Vector3(0, 0, cell_z/8),
                             size=mp.Vector3(cell_x - 2*pml_thickness, cell_y - 2*pml_thickness, 0)),
    "bottom": mp.FluxRegion(center=mp.Vector3(0, 0, -cell_z/8),
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

sources = [mp.Source(
    src=mp.ContinuousSource(frequency=omega, is_integrated=True),
    center=mp.Vector3(0, 0, cell_z/4),
    component=mp.Ex,
    size=mp.Vector3(cell_size.x, cell_size.y, 0),
    #size=mp.Vector3(cell_size.x - 2*pml_thickness, cell_size.y - 2*pml_thickness, 0)
)]


sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    boundary_layers=boundary_layers,
    sources=sources,
    default_material=material_function
)

sim.init_sim()

flux_monitors = {name: sim.add_flux(omega, 0, 1, region) for name, region in flux_regions.items()}
    
dft_fields = sim.add_dft_fields(
    [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz],
    [omega],
    where=mp.Volume(center=mp.Vector3(), size=sim.cell_size)
)

sim.load_minus_flux_data(flux_monitors["top"], box_pz_data_empty)
sim.load_minus_flux_data(flux_monitors["bottom"], box_mz_data_empty)
sim.load_minus_flux_data(flux_monitors["left"], box_mx_data_empty)
sim.load_minus_flux_data(flux_monitors["right"], box_px_data_empty)
sim.load_minus_flux_data(flux_monitors["front"], box_my_data_empty)
sim.load_minus_flux_data(flux_monitors["back"], box_py_data_empty)

sim.run(until=60)

flux_values = {name: mp.get_fluxes(mon)[0] for name, mon in flux_monitors.items()}
print("\n--- Flux Monitor Results ---")
for name, val in flux_values.items():
    print(f"{name.capitalize():<8}: {val: .4f}")
    flux_results[name].append(val)

# Save frequency-domain fields
field_data = {}
for comp in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
    mp_comp = getattr(mp, comp)
    array_freq = sim.get_dft_array(dft_fields, mp_comp, 0)
    field_data[comp] = array_freq

# Save all fields in a single .npz file
np.savez(f"SimData/fields.npz", **field_data)
print("💾 Saved combined E & H fields to SimData/fields.npz")
