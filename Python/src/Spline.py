import numpy as np
from scipy.interpolate import bisplev, bisplrep

class Spline:
    def __init__(self, Xfine, Yfine, Zfine, S = 0.5):
        print("Incoming shapes:", Xfine.shape, Yfine.shape, Zfine.shape)
        self.Xfine = Xfine
        self.Yfine = Yfine
        self.Zfine = Zfine

        
        self.tcks = bisplrep(Xfine.ravel(), Yfine.ravel(), Zfine.ravel(), s=S)
        self.max_curvature = self.__compute_max_mean_curvature__()

    def __evaluate_at_points__(self, x, y, dx = 0, dy = 0):
        return np.array(bisplev(x, y, self.tcks, dx = dx, dy = dy))
    
    def __compute_max_mean_curvature__(self):
        x = self.Xfine[0]
        y = self.Yfine[:, 0]

        fx = self.__evaluate_at_points__(x, y, dx = 1)
        fy = self.__evaluate_at_points__(x, y, dy = 1)
        fxx = self.__evaluate_at_points__(x, y, dx = 2)
        fyy = self.__evaluate_at_points__(x, y, dy = 2)
        fxy = self.__evaluate_at_points__(x, y, dx = 1, dy = 1)

        
        numer = (1 + fx**2) * fyy - 2 * fx * fy * fxy + (1 + fy**2) * fxx
        denom = (1 + fx**2 + fy**2)**1.5

        H = numer / (2 * denom)

        return np.max(abs(H))
    
    def calculate_points(self, resolution):
        x = np.linspace(self.Xfine.min(), self.Xfine.max(), resolution)
        y = np.linspace(self.Yfine.min(), self.Yfine.max(), resolution)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = self.__evaluate_at_points__(x, y)

        fx = self.__evaluate_at_points__(x, y, dx=1, dy=0)
        fy = self.__evaluate_at_points__(x, y, dx=0, dy=1)

        tau_x = np.stack([np.ones_like(fx), np.zeros_like(fx), fx], axis=-1)
        tau_y = np.stack([np.zeros_like(fy), np.ones_like(fy), fy], axis=-1)

        normals = np.cross(tau_x, tau_y)
        tau_y = np.cross(normals, tau_x)  # Ensure right-handed orthonormal basis

        # Normalize
        tau_x /= np.linalg.norm(tau_x, axis=-1, keepdims=True)
        tau_y /= np.linalg.norm(tau_y, axis=-1, keepdims=True)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        return points, normals.reshape(-1, 3), tau_x.reshape(-1, 3), tau_y.reshape(-1, 3)
