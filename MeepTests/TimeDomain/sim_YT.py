import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import json

# === Load surface parameters from JSON ===
with open("SurfaceData/surfaceParams11neg1.json", "r") as f:
    surface_params = json.load(f)

# === Simulation domain and resolution ===
Size_x = 2 * surface_params["halfWidth_x"]
Size_y = 2 * surface_params["halfWidth_y"]
Size_z = 2 * surface_params["halfWidth_z"]
cell_size = mp.Vector3(Size_x, Size_y, Size_z)

resolution = surface_params["resolution"]
pml_thickness = surface_params["pml_thickness"]
pmls = [mp.PML(thickness=pml_thickness)]

# === Bumpy surface geometry ===
def generate_bumpy_surface(surface_params):
    geometry = []
    bumpData = surface_params["bumpData"]
    epsilon1 = surface_params["epsilon1"]
    material = mp.Medium(epsilon=epsilon1)

    grid_size = 0.05
    x_vals = np.arange(-surface_params["halfWidth_x"], surface_params["halfWidth_x"], grid_size)
    y_vals = np.arange(-surface_params["halfWidth_y"], surface_params["halfWidth_y"], grid_size)

    for x in x_vals:
        for y in y_vals:
            z = 0.0
            for bump in bumpData:
                dx = x - bump["x0"]
                dy = y - bump["y0"]
                z += bump["height"] * np.exp(-(dx**2 + dy**2) / (2 * bump["sigma"]**2))
            if z > 0.01:
                geometry.append(mp.Block(
                    size=mp.Vector3(grid_size, grid_size, z),
                    center=mp.Vector3(x + grid_size/2, y + grid_size/2, z/2),
                    material=material
                ))
    return geometry

# === Source setup ===
center_wavelength = 0.55
min_wavelength = 0.4
max_wavelength = 0.7
center_freq = 1.0 / center_wavelength
freq_width = 1.0/min_wavelength - 1.0/max_wavelength

source_position = - Size_y / 2 + 0.3
gaussian_pulse = mp.GaussianSource(frequency=center_freq, fwidth=2*freq_width, is_integrated=True)
source = [mp.Source(gaussian_pulse,
                    component=mp.Ez,
                    center=mp.Vector3(0, source_position),
                    size=mp.Vector3(Size_x, 0, Size_z))]

# === Common frequency list ===
wavelengths = np.linspace(min_wavelength, max_wavelength, 41)
frequencies = 1.0 / wavelengths

# === Empty domain simulation ===
sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    sources=source,
                    boundary_layers=pmls,
                    geometry=[])

flux_size = min(Size_x, Size_y, Size_z) / 2
box_flux = sim.add_flux(center_freq, 0, 1,
                         mp.FluxRegion(center=mp.Vector3(y=Size_y/2 - 0.1), size=mp.Vector3(Size_x, 0, Size_z)))

sim.run(until_after_sources=mp.stop_when_fields_decayed(10, mp.Ez, mp.Vector3(0, Size_y/2 - 0.1), 1e-4))

incident_flux = mp.get_fluxes(box_flux)
flux_data = sim.get_flux_data(box_flux)

# === Bumpy surface simulation ===
sim.reset_meep()
geometry = generate_bumpy_surface(surface_params)

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    sources=source,
                    boundary_layers=pmls,
                    geometry=geometry)

box_flux_scat = sim.add_flux(center_freq, 0, 1,
                             mp.FluxRegion(center=mp.Vector3(y=Size_y/2 - 0.1), size=mp.Vector3(Size_x, 0, Size_z)))
sim.load_minus_flux_data(box_flux_scat, flux_data)

sim.run(until_after_sources=mp.stop_when_fields_decayed(10, mp.Ez, mp.Vector3(0, Size_y/2 - 0.1), 1e-4))

scattered_flux = mp.get_fluxes(box_flux_scat)
scattering_cross_section = np.array(scattered_flux) / np.array(incident_flux)

# === Output ===
plt.plot([center_wavelength], scattering_cross_section, 'ro')
plt.title("Scattering Cross Section at Center Wavelength")
plt.xlabel("Wavelength (um)")
plt.ylabel("Normalized Scattered Flux")
plt.grid(True)
plt.savefig("scattering_cross_section.png")
plt.close()

print("Scattering cross-section (normalized):", scattering_cross_section)
