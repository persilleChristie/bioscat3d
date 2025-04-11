import numpy as np

def dipole_field(r, theta):
    """
    Compute E and H fields for a z-directed Hertzian dipole at the origin.
    
    Parameters:
        r     : radial distance (meters)
        theta : polar angle (radians)
        f     : frequency (Hz)
        I0    : dipole current amplitude (A)
        l     : dipole length (m)
    
    Returns:
        E_r, E_theta, E_phi, H_r, H_theta, H_phi (complex values)
    """

    c = 3e8
    mu0 = 1.2566e-6
    eps0 = 8.854188e-12
    omega = 1
    k0 = 2 * np.pi / (325e-9)
    eta = np.sqrt(mu0 / eps0)
    I0l = 1.0

    # Common exponential factor
    exp_term = np.exp(-1j * k0 * r)

    # H field
    H_phi = 1j * k0 * I0l * np.sin(theta) / (4 * np.pi * r) * (1 + 1 / (1j * k0 * r)) * exp_term

    # E field
    E_r = eta * I0l * np.cos(theta) / (2 * np.pi * r**2) * (1 + 1 / (1j * k0 * r)) * exp_term
    E_theta = 1j * k0 * eta * I0l * np.sin(theta) / (4 * np.pi * r) * (1 + 1 / (1j * k0 * r) - 1 / (k0 * r)**2) * exp_term

    # All other components are zero
    E_phi = 0
    H_r = 0
    H_theta = 0

    return E_r, E_theta, E_phi, H_r, H_theta, H_phi

r = np.sqrt((-1.885506218)**2 + 9.645395422**2 + 5.423210620**2)
theta = np.arccos(5.423210620/r)
r = np.sqrt((-1.885506218)**2 + 9.645395422**2 + 5.423210620**2)
theta = np.arccos(5.423210620/r)

E_r, E_theta, E_phi, H_r, H_theta, H_phi = dipole_field(r, theta)

print(f"E_r      = {E_r}")
print(f"E_theta  = {E_theta}")
print(f"H_phi    = {H_phi}")

import numpy as np

# Combine E and H into full vectors
E_vec = np.array([E_r, E_theta, E_phi], dtype=complex)
H_vec = np.array([H_r, H_theta, H_phi], dtype=complex)

# Compute magnitudes (Euclidean norm)
norm_E = np.linalg.norm(E_vec)
norm_H = np.linalg.norm(H_vec)


# Avoid division by zero
if norm_H != 0:
    ratio = norm_E / norm_H
else:
    ratio = np.inf

print(f"‖E‖ = {norm_E:.3e}")
print(f"‖H‖ = {norm_H:.3e}")
print(f"‖E‖ / ‖H‖ = {ratio:.3e}")
