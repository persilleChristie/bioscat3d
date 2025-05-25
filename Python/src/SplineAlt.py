import numpy as np
from scipy.interpolate import bisplev, bisplrep

class Spline:
    def __init__(self, Xfine, Yfine, Zfine):
        self.Xfine = Xfine
        self.Yfine = Yfine
        self.Zfine = Zfine

        self.tcks = bisplrep(Xfine.ravel(), Yfine.ravel(), Zfine.ravel(), s=0.5)
        self.max_curvature = self._compute_max_mean_curvature()

    def _evaluate_at_points(self, x, y, dx=0, dy=0):
        return np.array(bisplev(x, y, self.tcks, dx=dx, dy=dy))

    def _compute_max_mean_curvature(self):
        x = np.linspace(self.Xfine.min(), self.Xfine.max(), 100)
        y = np.linspace(self.Yfine.min(), self.Yfine.max(), 100)

        fx = self._evaluate_at_points(x, y, dx=1, dy=0)
        fy = self._evaluate_at_points(x, y, dx=0, dy=1)
        fxx = self._evaluate_at_points(x, y, dx=2, dy=0)
        fyy = self._evaluate_at_points(x, y, dx=0, dy=2)
        fxy = self._evaluate_at_points(x, y, dx=1, dy=1)

        numer = (1 + fx**2) * fyy - 2 * fx * fy * fxy + (1 + fy**2) * fxx
        denom = (1 + fx**2 + fy**2)**1.5
        H = numer / (2 * denom)

        return np.max(np.abs(H))

    def calculate_points(self, resolution):
        # --- Meshgrid in parameter space ---
        x = np.linspace(self.Xfine.min(), self.Xfine.max(), resolution)
        y = np.linspace(self.Yfine.min(), self.Yfine.max(), resolution)
        X, Y = np.meshgrid(x, y, indexing='ij')  # indexing='ij' ensures correct flattening

        # --- Evaluate spline and gradients ---
        Z = self._evaluate_at_points(x, y)
        fx = self._evaluate_at_points(x, y, dx=1, dy=0)
        fy = self._evaluate_at_points(x, y, dx=0, dy=1)

        # --- Tangents (parametric surface r(x, y) = (x, y, f(x,y)) ---
        tau1 = np.stack([np.ones_like(fx), np.zeros_like(fx), fx], axis=-1)
        tau2 = np.stack([np.zeros_like(fy), np.ones_like(fy), fy], axis=-1)

        # --- Normals and orthonormalization ---
        normals = np.cross(tau1, tau2)
        tau2 = np.cross(normals, tau1)

        # --- Normalize all vectors ---
        tau1 /= np.linalg.norm(tau1, axis=-1, keepdims=True)
        tau2 /= np.linalg.norm(tau2, axis=-1, keepdims=True)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        # --- Flatten for output ---
        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        tau1 = tau1.reshape(-1, 3)
        tau2 = tau2.reshape(-1, 3)
        normals = normals.reshape(-1, 3)

        # --- Optional orthogonality check ---
        dot_12 = np.einsum('ij,ij->i', tau1, tau2)
        dot_1n = np.einsum('ij,ij->i', tau1, normals)
        dot_2n = np.einsum('ij,ij->i', tau2, normals)
        max_dev = max(np.abs(dot_12).max(), np.abs(dot_1n).max(), np.abs(dot_2n).max())
        if max_dev > 1e-10:
            print(f"⚠️ Orthogonality deviation: {max_dev:.2e}")

        return points, normals, tau1, tau2
