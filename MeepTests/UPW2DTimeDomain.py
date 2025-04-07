import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# --- Simulation parameters ---
resolution = 50
cell_size = mp.Vector3(16, 12, 0)

fcen = 1.0
df = 0.2

sources = [mp.Source(
    src=mp.GaussianSource(frequency=fcen, fwidth=df),
    component=mp.Ez,
    center=mp.Vector3(-6, 0),           # centered in y
    size=mp.Vector3(0, 12)              # full y-span to make it uniform
)]

pml_layers = [mp.PML(1.0)]

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    sources=sources,
    boundary_layers=pml_layers,
    geometry=[],
    dimensions=2
)

# --- Sample field along x at y=0 ---
x_span = 12
npts = int(x_span * resolution)
ez_data = []

def get_ez(sim):
    line = sim.get_array(center=mp.Vector3(0, 0), size=mp.Vector3(x_span, 0), component=mp.Ez)
    ez_data.append(np.copy(line))

# --- Run simulation ---
sim.run(mp.at_every(0.2, get_ez), until=200)

# --- Plot result ---
ez_array = np.array(ez_data).T
ez_array /= np.max(np.abs(ez_array))  # normalize

plt.figure(figsize=(10, 5))
plt.imshow(ez_array, aspect='auto', origin='lower',
           extent=[0, sim.meep_time(), -x_span/2, x_span/2],
           cmap='viridis')
plt.xlabel("Time (a.u.)")
plt.ylabel("Position x (μm)")
plt.title("Ez field (x vs time) for 2D UPW")
plt.colorbar(label="Normalized Ez")
plt.tight_layout()
plt.savefig("UPW_2D_fixed.png")
print("✅ Saved corrected field plot to UPW_2D_fixed.png")
