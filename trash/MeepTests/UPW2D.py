import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
lambda0 = 1.0         # wavelength in μm (arbitrary)
fcen = 1 / lambda0    # frequency
dpml = 1.0            # PML thickness
sx = 32                # total size in x
sy = 16                # total size in y
resolution = 20       # pixels per μm

cell = mp.Vector3(sx, sy)
pml_layers = [mp.PML(dpml)]

# === Eigenmode source ===
sources = [mp.EigenModeSource(
               src=mp.ContinuousSource(frequency=fcen),
               center=mp.Vector3(-0.5*sx + dpml + 1.0),  # safely inside domain
               size=mp.Vector3(0, sy),
               direction=mp.X,
               eig_match_freq=True,
               #eig_amplitude=1.0,
               component=mp.Ez)]

# === Simulation ===
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=[],
                    sources=sources,
                    resolution=resolution,
                    force_complex_fields=True)

sim.init_sim()
sim.solve_cw(tol=1e-8, maxiters=10000)

# === Get field ===
ez = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

# === Plot ===
plt.figure(figsize=(10, 4))
plt.imshow(np.real(ez).T, interpolation='spline36', cmap='RdBu',
           extent=[-sx/2, sx/2, -sy/2, sy/2])
plt.title(f"Re(Ez) — Plane Wave via EigenModeSource")
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")
plt.colorbar(label="Re(Ez)")
plt.show(block=False)
input("✅ Press Enter to exit...")
