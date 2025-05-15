'''
Hertzian dipole class used for for approximation of the scattered electric and magnetic field
'''

#Import of modules
import numpy as np
import warnings
import multiprocessing
import pandas as pd

import numpy as np
from numba import njit, prange

# 1. The Numba‐jitted core routine (no Python‐style checks, uses explicit loops)
@njit(parallel=True, fastmath=True)
def _evaluate_hertzian_fields(positions, directions, mu, epsilon, omega, X):
    """
    positions : (M,3) float64
    directions: (M,3) float64, assumed unit‐norm
    X         : (N,3) float64
    returns   : (2, M, N, 3) complex128 array
    """
    M = positions.shape[0]
    N = X.shape[0]

    # precompute common scalars
    k = omega * np.sqrt(epsilon * mu)
    xi = 1j * omega * mu / (4.0 * np.pi)

    # allocate output
    fields = np.empty((2, M, N, 3), dtype=np.complex128)

    # main double‐loop over dipoles i and points j
    for i in prange(M):
        px, py, pz = positions[i, 0], positions[i, 1], positions[i, 2]
        dx, dy, dz = directions[i, 0], directions[i, 1], directions[i, 2]

        for j in range(N):
            # displacement vector
            x = X[j, 0] - px
            y = X[j, 1] - py
            z = X[j, 2] - pz

            # radial distance
            r2 = x*x + y*y + z*z
            if r2 == 0.0:
                r2 = 1e-32
            r = np.sqrt(r2)

            # inverse powers of r
            inv_r  = 1.0 / r
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r2 * inv_r
            inv_r4 = inv_r3 * inv_r
            inv_r5 = inv_r4 * inv_r

            # projection of direction on r‐vector
            dotted = dx*x + dy*y + dz*z

            # common field factors
            R = inv_r3 + 1j * k * inv_r2
            Phi_x = 3*x*inv_r5  + 3j*k*x*inv_r4  - (k*k)*x*inv_r3
            Phi_y = 3*y*inv_r5  + 3j*k*y*inv_r4  - (k*k)*y*inv_r3
            Phi_z = 3*z*inv_r5  + 3j*k*z*inv_r4  - (k*k)*z*inv_r3
            phase = np.exp(-1j * k * r)

            inv_k2 = 1.0 / (k*k)
            term1 = xi * inv_k2 * R
            term2 = xi * inv_r

            # Electric field components
            Ex = (dx*(term1 - term2) - xi*inv_k2 * Phi_x*dotted) * phase
            Ey = (dy*(term1 - term2) - xi*inv_k2 * Phi_y*dotted) * phase
            Ez = (dz*(term1 - term2) - xi*inv_k2 * Phi_z*dotted) * phase

            # Magnetic field components (approximate form)
            factor = 1.0 / (4.0 * np.pi)
            Hx = factor * ((dy*z - dz*y) * R) * phase
            Hy = factor * ((dz*x - dx*z) * R) * phase
            Hz = factor * ((dx*y - dy*x) * R) * phase

            # store
            fields[0, i, j, 0] = Ex
            fields[0, i, j, 1] = Ey
            fields[0, i, j, 2] = Ez
            fields[1, i, j, 0] = Hx
            fields[1, i, j, 1] = Hy
            fields[1, i, j, 2] = Hz

    return fields


# 2. A thin Python wrapper for validation + storage:
class Hertzian_Dipole:
    def __init__(self, positions, directions, mu, epsilon, omega):
        # basic Python‐side checks and normalization:
        positions = np.asarray(positions, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must be (M,3)")
        if directions.ndim != 2 or directions.shape != positions.shape:
            raise ValueError("directions must be (M,3)")

        norms = np.linalg.norm(directions, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-1):
            raise ValueError("direction vectors must be unit‐norm")

        for name, val in (("mu",mu),("epsilon",epsilon),("omega",omega)):
            if not np.isscalar(val):
                raise TypeError(f"{name} must be a scalar")

        # store
        self.positions = positions
        self.directions = directions
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.omega = float(omega)

    def evaluate_at_points(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("X must be (N,3)")
        return _evaluate_hertzian_fields(
            self.positions,
            self.directions,
            self.mu,
            self.epsilon,
            self.omega,
            X
        )

import ast  # For safely evaluating string representations of lists

def compute_fields_from_csv(param_file, testpoints_file, output_file):
    # Read parameters
    param_df = pd.read_csv(param_file)
    params = dict(zip(param_df["Parameter"], param_df["Value"]))

    # Convert values properly
    mu = float(params["mu"])
    epsilon = float(params["epsilon"])
    omega = float(params["omega"])
    position = np.array([params["position_x"], params["position_y"], params["position_z"]])
    direction = np.array([params["direction_x"], params["direction_y"], params["direction_z"]])

    # Read test points
    testpoints_df = pd.read_csv(testpoints_file)
    testpoints = testpoints_df.to_numpy()  # Convert DataFrame to NumPy array

    # Debugging: Check types
    print(f"mu: {mu}, epsilon: {epsilon}, omega: {omega}")
    print(f"Position: {position}, Direction: {direction}")
    print(f"Testpoints shape: {testpoints.shape}")

    # Compute fields (assuming Hertzian_Dipole is defined)
    DP = Hertzian_Dipole(np.array([position]), np.array([direction]), mu, epsilon, omega)
    E, H = DP.evaluate_at_points(testpoints)
    E ,H = E[0], H[0]
    print(f"Calculated impedance: {np.linalg.norm(E,axis=1)/np.linalg.norm(H,axis=1)}")

    # Convert complex values into real & imaginary parts for saving
    data = {
        "Ex_Re": E[:, 0].real, "Ex_Im": E[:, 0].imag,
        "Ey_Re": E[:, 1].real, "Ey_Im": E[:, 1].imag,
        "Ez_Re": E[:, 2].real, "Ez_Im": E[:, 2].imag,
        "Hx_Re": H[:, 0].real, "Hx_Im": H[:, 0].imag,
        "Hy_Re": H[:, 1].real, "Hy_Im": H[:, 1].imag,
        "Hz_Re": H[:, 2].real, "Hz_Im": H[:, 2].imag,
    }

    output_df = pd.DataFrame(data)
    output_df.to_csv(output_file, index=False)

    print(f"Computed field data saved to {output_file}")
    