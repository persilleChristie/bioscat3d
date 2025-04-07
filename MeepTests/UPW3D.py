import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
lambda0 = 1.0         # wavelength in μm
fcen = 1 / lambda0
resolution = 20       # pixels/μm

dpml = 1.0            # PML thickness
sx = 8                # domain size (x)
sy = 4                # domain size (y)
sz = 4                # domain size (z)

cell = mp.Vector3(sx, sy, sz)
pml_layers = [mp.PML(dpml)]

# === EigenModeSource (inject plane wave in +x direction) ===
sources = [mp.EigenModeSource(
    src=mp.ContinuousSource(frequency=fcen),
    center=mp.Vector3(-0.5 * sx + dpml + 1.0),
    size=mp.Vector3(0, sy, sz),
    direction=mp.X,
    eig_match_freq=True,
    component=mp.Ez  # TM polarization
)]

# === Simulation ===
sim = mp.Simulation(cell_size=cell,
                    resolution=resolution,
                    boundary_layers=pml_layers,
                    geometry=[],
                    sources=sources,
                    dimensions=3,
                    force_complex_fields=True)

# === Solve ===
sim.init_sim()
sim.solve_cw(tol=1e-8, maxiters=10000)

# === Extract a 2D slice of Ez (x-y at mid-z) ===
ez = sim.get_array(center=mp.Vector3(z=0), size=mp.Vector3(sx, sy, 0), component=mp.Ez)

# === Plot ===
plt.figure(figsize=(10, 4))
plt.imshow(np.real(ez).T, interpolation='spline36', cmap='RdBu',
           extent=[-sx/2, sx/2, -sy/2, sy/2])
plt.title("Re(Ez) in x–y plane (z = 0)")
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")
plt.colorbar(label="Re(Ez)")
plt.show(block=False)
input("✅ Press Enter to close and exit...")

