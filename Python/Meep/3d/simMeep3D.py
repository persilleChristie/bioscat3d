import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import math

keyword = "Meep"
path = "../../../json/surfaceParams" + keyword + ".json"
def load_params(path = path):
    with open(path, "r") as f:
        return json.load(f)
params = load_params()

# Cell and resolution
pml_thickness = params["pml_thickness"]
monitor_size = params["monitor_size"]
half_x = params["halfWidth_x"]
half_y = params["halfWidth_y"]
half_z = params["halfWidth_z"]
cell_size = mp.Vector3(2*half_x + 2*pml_thickness,
                       2*half_y + 2*pml_thickness,
                       2*half_z + 2*pml_thickness)

#omega = params["omega"]
betas = params["betas"]
# Material
epsilon1 = params["epsilon1"] 
substrate = mp.Medium(epsilon = epsilon1)
bumps = params["bumpData"]

## Must be at least 8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
resolution = params["resolution"]

# fcen = params["freqCenter"]
# df = params["freqWidth"]
# nfrq = params["freqs_n"]

lambda_min = params["minLambda"]
lambda_max = params["maxLambda"]
lambda_n = params["lambda_n"]


fcen = 1/((lambda_max-lambda_min)/2)
df = lambda_max-lambda_min
nfrq = lambda_n


pml_layers = [mp.PML(thickness=pml_thickness)] 

def make_source(component):
    return [mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True),
        center=mp.Vector3(0, 0, half_z - 0.5),
        size=mp.Vector3(cell_size[0], cell_size[1], 0),
        component=component
    )]


def run_meep_simulation(sources):
    # Reference simulation (no bumps)
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=mp.Vector3(),
    )

    box1_x1 = sim.add_flux(fcen, df, nfrq, mp.FluxRegion(center=mp.Vector3(x=-monitor_size/2), size=mp.Vector3(0, monitor_size, monitor_size)))
    box1_x2 = sim.add_flux(fcen, df, nfrq, mp.FluxRegion(center=mp.Vector3(x=+monitor_size/2), size=mp.Vector3(0, monitor_size, monitor_size)))
    box1_y1 = sim.add_flux(fcen, df, nfrq, mp.FluxRegion(center=mp.Vector3(y=-monitor_size/2), size=mp.Vector3(monitor_size, 0, monitor_size)))
    box1_y2 = sim.add_flux(fcen, df, nfrq, mp.FluxRegion(center=mp.Vector3(y=+monitor_size/2), size=mp.Vector3(monitor_size, 0, monitor_size)))
    box1_z1 = sim.add_flux(fcen, df, nfrq, mp.FluxRegion(center=mp.Vector3(z=-monitor_size/2), size=mp.Vector3(monitor_size, monitor_size, 0)))
    box1_z2 = sim.add_flux(fcen, df, nfrq, mp.FluxRegion(center=mp.Vector3(z=+monitor_size/2), size=mp.Vector3(monitor_size, monitor_size, 0)))
    sim.run(until_after_sources=60)

    box_x1 = mp.get_fluxes(box1_x1)
    box_x2 = mp.get_fluxes(box1_x2)
    box_y1 = mp.get_fluxes(box1_y1)
    box_y2 = mp.get_fluxes(box1_y2)
    box_z1 = mp.get_fluxes(box1_z1)
    box_z2 = mp.get_fluxes(box1_z2)

    # Save individual components for later printing/plotting
    inc_flux_components = {
        "X1": np.asarray(box_x1),
        "X2": np.asarray(box_x2),
        "Y1": np.asarray(box_y1),
        "Y2": np.asarray(box_y2),
        "Z1": np.asarray(box_z1),
        "Z2": np.asarray(box_z2),
    }

    box_x1_data = sim.get_flux_data(box1_x1)
    box_x2_data = sim.get_flux_data(box1_x2)
    box_y1_data = sim.get_flux_data(box1_y1)
    box_y2_data = sim.get_flux_data(box1_y2)
    box_z1_data = sim.get_flux_data(box1_z1)
    box_z2_data = sim.get_flux_data(box1_z2)

    sim.reset_meep()

# Structured simulation (with bumps) and edges
    def material_function(p):
        if abs(p.x) >= 1 or abs(p.y) >= 1:
            return mp.air
        z_surface = 0
        for b in bumps:
            dx = p.x - b["x0"]
            dy = p.y - b["y0"]
            exponent = - (dx**2 + dy**2) / (2 * b["sigma"]**2)
            z_surface += b["height"] * math.exp(exponent)
        return substrate if p.z + 1.5 < z_surface else mp.air



    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=mp.Vector3(),
        default_material=material_function,
    )

    box2_x1 = sim.add_flux(fcen, df, nfrq, mp.FluxRegion(center=mp.Vector3(x=-monitor_size/2), size=mp.Vector3(0, monitor_size, monitor_size)))
    box2_x2 = sim.add_flux(fcen, df, nfrq, mp.FluxRegion(center=mp.Vector3(x=+monitor_size/2), size=mp.Vector3(0, monitor_size, monitor_size)))
    box2_y1 = sim.add_flux(fcen, df, nfrq, mp.FluxRegion(center=mp.Vector3(y=-monitor_size/2), size=mp.Vector3(monitor_size, 0, monitor_size)))
    box2_y2 = sim.add_flux(fcen, df, nfrq, mp.FluxRegion(center=mp.Vector3(y=+monitor_size/2), size=mp.Vector3(monitor_size, 0, monitor_size)))
    box2_z1 = sim.add_flux(fcen, df, nfrq, mp.FluxRegion(center=mp.Vector3(z=-monitor_size/2), size=mp.Vector3(monitor_size, monitor_size, 0)))
    box2_z2 = sim.add_flux(fcen, df, nfrq, mp.FluxRegion(center=mp.Vector3(z=+monitor_size/2), size=mp.Vector3(monitor_size, monitor_size, 0)))

    sim.load_minus_flux_data(box2_x1, box_x1_data)
    sim.load_minus_flux_data(box2_x2, box_x2_data)
    sim.load_minus_flux_data(box2_y1, box_y1_data)
    sim.load_minus_flux_data(box2_y2, box_y2_data)
    sim.load_minus_flux_data(box2_z1, box_z1_data)
    sim.load_minus_flux_data(box2_z2, box_z2_data)

    sim.run(until_after_sources=60) # maybe 20 is too little? 

    box_x1_flux = mp.get_fluxes(box2_x1)
    box_x2_flux = mp.get_fluxes(box2_x2)
    box_y1_flux = mp.get_fluxes(box2_y1)
    box_y2_flux = mp.get_fluxes(box2_y2)
    box_z1_flux = mp.get_fluxes(box2_z1)
    box_z2_flux = mp.get_fluxes(box2_z2)

    # Save individual components for later printing/plotting
    scatt_flux_components = {
        "X1": np.asarray(box_x1_flux),
        "X2": np.asarray(box_x2_flux),
        "Y1": np.asarray(box_y1_flux),
        "Y2": np.asarray(box_y2_flux),
        "Z1": np.asarray(box_z1_flux),
        "Z2": np.asarray(box_z2_flux),
    }


    freqs = mp.get_flux_freqs(box1_x1)  # or any box that was added with add_flux
    return scatt_flux_components, inc_flux_components, freqs


flux_results = []
inc_flux_results = []

def print_parameters():
    print("Simulation Parameters:")
    print(f"  resolution      = {resolution}")
    print(f"  cell_size       = ({cell_size.x}, {cell_size.y}, {cell_size.z})")
    print(f"  PML thickness   = {pml_thickness}")
    print(f"  monitor size    = {monitor_size}")
    print(f"  substrate ε     = {epsilon1}")
    print(f"  number of bumps = {len(bumps)}")
    print(f"  freq center     = {fcen:.4f} (→ λ ≈ {100 / fcen:.1f} nm)")
    print(f"  freq width      = {df:.4f} (from {fcen - df/2:.4f} to {fcen + df/2:.4f})")
    print(f"  freqs_n         = {nfrq} points")
    print("  betas (deg)     =", [round(np.degrees(b), 2) for b in betas])
    print()
print_parameters()


for beta in betas:
    print("beta = " + str(beta))
    cos2 = np.cos(beta)**2
    sin2 = np.sin(beta)**2

    sources_1 = make_source(mp.Ey)
    sources_2 = make_source(mp.Ex)
    
    flux_1, inc_flux_1, freqs = run_meep_simulation(sources_1)
    flux_2, inc_flux_2, freqs = run_meep_simulation(sources_2)

    for direction in ["X1", "X2", "Y1", "Y2", "Z1", "Z2"]:
        flux_arr = cos2 * flux_1[direction] + sin2 * flux_2[direction]
        inc_flux_arr = cos2 * inc_flux_1[direction] + sin2 * inc_flux_2[direction]

        for f, scatt_val, inc_val in zip(freqs, flux_arr, inc_flux_arr):
            flux_results.append({
                "frequency": f,
                "beta": np.degrees(beta),
                "direction": direction,
                "flux": scatt_val,
                "type": "scattered"
            })
            inc_flux_results.append({
                "frequency": f,
                "beta": np.degrees(beta),
                "direction": direction,
                "flux": inc_val,
                "type": "incident"
            })