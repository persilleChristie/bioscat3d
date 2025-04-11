import meep as mp
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# === Geometry parameters ===
n = 3.4
w = 1
r = 1
pad = 4
dpml = 2

sxy = 2 * (r + w + pad + dpml)
cell_size = mp.Vector3(sxy, sxy)

pml_layers = [mp.PML(dpml)]

nonpml_vol = mp.Volume(mp.Vector3(), size=mp.Vector3(sxy - 2 * dpml, sxy - 2 * dpml))

# === Geometry: Optional cylinder (comment out if not needed) ===
geometry = [mp.Cylinder(radius=r + w, material=mp.Medium(index=n)),
            mp.Cylinder(radius=r)]

# === Frequency ===
fcen = 0.118

# === Single point source (Ez polarized) ===
src = [mp.Source(mp.ContinuousSource(fcen),
                 component=mp.Ez,
                 center=mp.Vector3(0.5))]  # Slight offset from center

# === Simulation Setup ===
sim = mp.Simulation(cell_size=cell_size,
                    geometry=geometry,
                    sources=src,
                    resolution=10,
                    force_complex_fields=True,
                    boundary_layers=pml_layers)

# === Convergence test ===
num_tols = 5
tols = np.power(10, np.arange(-8.0, -8.0 - num_tols, -1.0))
ez_dat = np.zeros((122, 122, num_tols), dtype=np.complex_)

for i in range(num_tols):
    sim.init_sim()
    sim.solve_cw(tols[i], 10000, 10)
    ez_dat[:, :, i] = sim.get_array(vol=nonpml_vol, component=mp.Ez)

err_dat = np.zeros(num_tols - 1)
for i in range(num_tols - 1):
    err_dat[i] = LA.norm(ez_dat[:, :, i] - ez_dat[:, :, num_tols - 1])

# === Plot convergence ===
plt.figure(dpi=150)
plt.loglog(tols[:num_tols - 1], err_dat, 'bo-')
plt.xlabel("frequency-domain solver tolerance")
plt.ylabel("L2 norm of error in fields")
plt.title("Convergence of Ez field with tolerance")
plt.show(block=False)
input("✅ Press Enter to view field...")

# === Plot dielectric + field ===
eps_data = sim.get_array(vol=nonpml_vol, component=mp.Dielectric)
ez_data = np.real(ez_dat[:, :, num_tols - 1])

plt.figure()
plt.imshow(eps_data.T, interpolation='spline36', cmap='binary')
plt.imshow(ez_data.T, interpolation='spline36', cmap='RdBu', alpha=0.9)
plt.title("Real(Ez) field from point source")
plt.axis('off')
plt.show(block=False)
input("✅ Done. Press Enter to exit...")

# === Check convergence trend ===
if np.all(np.diff(err_dat) < 0):
    print("PASSED solve_cw test: error in the fields is decreasing with increasing resolution")
else:
    print("FAILED solve_cw test: error in the fields is NOT decreasing with increasing resolution")
