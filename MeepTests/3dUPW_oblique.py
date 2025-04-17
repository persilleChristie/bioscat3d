import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import cmath
import plotly.graph_objects as go


# === Parameters ===
lambda0 = 325e-3        # wavelength (μm)
fcen = 1 / lambda0      # frequency
theta = 30              # angle of incidence (degrees) in x–z plane

resolution = 10         # pixels/μm

dpml = 1.0              # PML thickness
sx = 3
sy = 3
sz = 3
cell = mp.Vector3(sx, sy, sz)
pml_layers = [mp.PML(dpml)]

# === Wavevector Components (for amp_func) ===
theta_rad = np.radians(theta)
k = 2 * np.pi * fcen
kx = k * np.sin(theta_rad)
kz = k * np.cos(theta_rad)

def oblique_amp_func(pos):
    return cmath.exp(1j * (kx * pos.x + kz * pos.z))

# === Plane Wave Source ===
sources = [mp.Source(
    src=mp.ContinuousSource(frequency=fcen),
    component=mp.Ey,
    center=mp.Vector3(-0.5 * sx + dpml + 0.1),
    size=mp.Vector3(0, sy, sz),  # source is y–z plane at x = const
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

# === Solve in frequency domain ===
sim.init_sim()
sim.solve_cw(tol=1e-4)

# === Extract a 2D slice of Ey (x–z at y = 0) ===
ey = sim.get_array(center=mp.Vector3(y=0), size=mp.Vector3(sx, 0, sz), component=mp.Ey)

# === Plot: x–z plane slice ===
plt.figure(figsize=(10, 4))
plt.imshow(np.real(ey).T, interpolation='spline36', cmap='RdBu',
           extent=[-sx/2, sx/2, -sz/2, sz/2])
plt.title(f"Re(Ey) in x–z plane (y = 0), Oblique Incidence θ = {theta}°")
plt.xlabel("x (μm)")
plt.ylabel("z (μm)")
plt.colorbar(label="Re(Ey)")
plt.show(block=False)
input("✅ Press Enter to exit...")


def save_field_surface(field, x_range, y_range, title, filename):
    nx, ny = field.shape
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    fig = go.Figure(data=[go.Surface(z=np.real(field), x=X, y=Y, colorscale='RdBu')])

    fig.update_layout(
        scene=dict(
            xaxis_title="x (μm)",
            yaxis_title="y (μm)",
            zaxis_title="Re(Field)",
            aspectmode='data'
        ),
        title=title
    )

    fig.write_html(filename)
    print(f"✅ Field slice saved to: {filename}")

# === Save interactive 3D surface plot (Plotly)
save_field_surface(
    field=ey.T,
    x_range=[-sx/2, sx/2],
    y_range=[-sy/2, sy/2],
    title=f"Re(Ey) in x–y plane (z = 0), Oblique Incidence θ = {theta}°",
    filename="field_slice_ey_oblique.html"
)

def save_vector_field_plot(Ex, Ey, Ez, extent, spacing=3, filename="vector_field.html"):
    import plotly.graph_objects as go
    sx, sy, sz = extent

    # Downsample
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
        title="3D Vector Field (E-field)",
    )
    fig.write_html(filename)
    print(f"✅ Saved 3D vector field to: {filename}")


# After solve_cw
Ex = sim.get_array(center=mp.Vector3(), size=mp.Vector3(sx, sy, sz), component=mp.Ex)
Ey = sim.get_array(center=mp.Vector3(), size=mp.Vector3(sx, sy, sz), component=mp.Ey)
Ez = sim.get_array(center=mp.Vector3(), size=mp.Vector3(sx, sy, sz), component=mp.Ez)

save_vector_field_plot(Ex, Ey, Ez, extent=(sx, sy, sz), spacing=3, filename="3D_vector_field.html")
