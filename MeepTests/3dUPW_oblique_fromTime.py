import meep as mp
import numpy as np
import cmath
import plotly.graph_objects as go

# === Parameters ===
lambda0 = 325e-3            # wavelength (μm)
fcen = 1 / lambda0          # frequency (1/μm)
theta = 0                  # angle of incidence (degrees)

resolution = 10             # pixels/μm

dpml = 1.0                  # PML thickness
sx = 3
sy = 3
sz = 3
cell = mp.Vector3(sx, sy, sz)
pml_layers = [mp.PML(dpml)]

# === Wavevector for amp_func ===
theta_rad = np.radians(theta)
k = 2 * np.pi * fcen
kx = k * np.sin(theta_rad)
kz = k * np.cos(theta_rad)

def oblique_amp_func(pos):
    return cmath.exp(1j * (kx * pos.x + kz * pos.z))

# === Plane Wave Source (Gaussian envelope helps convergence)
sources = [mp.Source(
    src=mp.GaussianSource(frequency=fcen, fwidth=fcen / 10),
    component=mp.Ey,
    center=mp.Vector3(-0.5 * sx + dpml + 0.1),
    size=mp.Vector3(0, sy, sz),
    amp_func=oblique_amp_func
)]

# === Simulation ===
sim = mp.Simulation(
    cell_size=cell,
    resolution=resolution,
    boundary_layers=pml_layers,
    sources=sources,
    geometry=[],
    dimensions=3,
    force_complex_fields=True
)

# === DFT monitor for frequency-domain field ===
dft_obj = sim.add_dft_fields(
    [mp.Ex, mp.Ey, mp.Ez],
    fcen,
    0,     # no frequency width — single frequency
    1,     # just one frequency point
    center=mp.Vector3(),
    size=mp.Vector3(sx, sy, sz)
)

sim.run(until=200)


# === Extract frequency-domain E-fields
Ex = sim.get_dft_array(dft_obj, mp.Ex, 0)
Ey = sim.get_dft_array(dft_obj, mp.Ey, 0)
Ez = sim.get_dft_array(dft_obj, mp.Ez, 0)

# === Plot 3D vector field with Plotly
def save_vector_field_plot(Ex, Ey, Ez, extent, spacing=3, filename="3D_vector_field_dft.html"):
    sx, sy, sz = extent

    # Downsample for performance
    Ex_ds = Ex[::spacing, ::spacing, ::spacing]
    Ey_ds = Ey[::spacing, ::spacing, ::spacing]
    Ez_ds = Ez[::spacing, ::spacing, ::spacing]

    nx, ny, nz = Ex_ds.shape
    x = np.linspace(-sx/2, sx/2, nx)
    y = np.linspace(-sy/2, sy/2, ny)
    z = np.linspace(-sz/2, sz/2, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    fig = go.Figure(data=go.Cone(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        u=np.real(Ex_ds).flatten(),
        v=np.real(Ey_ds).flatten(),
        w=np.real(Ez_ds).flatten(),
        sizemode="absolute",
        sizeref=2,
        anchor="tail",
        colorscale='Bluered'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='cube'
        ),
        title=f"3D Vector Field (E-field) at f = {fcen:.2f} μm⁻¹",
    )

    fig.write_html(filename)
    print(f"✅ Saved interactive 3D vector field: {filename}")

# === Save the 3D plot
save_vector_field_plot(Ex, Ey, Ez, extent=(sx, sy, sz), spacing=3, filename="3D_vector_field_dft.html")
