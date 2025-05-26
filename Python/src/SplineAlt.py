import numpy as np
import pandas as pd
from scipy.interpolate import bisplev, bisplrep


class Spline:
    def __init__(self, Xfine, Yfine, Zfine, smoothing=5):
        self.Xfine = Xfine
        self.Yfine = Yfine
        self.Zfine = Zfine
        self.tcks = bisplrep(Xfine.ravel(), Yfine.ravel(), Zfine.ravel(), s=smoothing)
        self.max_curvature = self.__compute_max_mean_curvature__()

    def __evaluate_at_points__(self, x, y, dx=0, dy=0):
        return np.array(bisplev(x, y, self.tcks, dx=dx, dy=dy))

    def __compute_max_mean_curvature__(self):
        x = self.Xfine[0]
        y = self.Yfine[:, 0]

        fx = self.__evaluate_at_points__(x, y, dx=1)
        fy = self.__evaluate_at_points__(x, y, dy=1)
        fxx = self.__evaluate_at_points__(x, y, dx=2)
        fyy = self.__evaluate_at_points__(x, y, dy=2)
        fxy = self.__evaluate_at_points__(x, y, dx=1, dy=1)

        numer = (1 + fx**2) * fyy - 2 * fx * fy * fxy + (1 + fy**2) * fxx
        denom = (1 + fx**2 + fy**2)**1.5
        H = numer / (2 * denom)

        return np.max(np.abs(H))

    def __get_stable_reference__(self, normals):
        abs_n = np.abs(normals)
        ref = np.zeros_like(normals)
        idx = np.argmin(abs_n, axis=1)
        ref[np.arange(len(normals)), idx] = 1
        return ref

    def __calculate_normals_tangents__(self, x, y):
        fx = self.__evaluate_at_points__(x, y, dx=1).ravel()
        fy = self.__evaluate_at_points__(x, y, dy=1).ravel()

        tangent1 = np.stack([np.ones_like(fx), np.zeros_like(fx), fx], axis=-1)
        tangent2 = np.stack([np.zeros_like(fy), np.ones_like(fy), fy], axis=-1)

        normals = np.cross(tangent1, tangent2)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        ref = self.__get_stable_reference__(normals)
        dot_ref_n = np.einsum('ij,ij->i', ref, normals)[:, np.newaxis]
        tangent1 = ref - dot_ref_n * normals
        tangent1 /= np.linalg.norm(tangent1, axis=1, keepdims=True)

        tangent2 = np.cross(normals, tangent1)
        tangent2 /= np.linalg.norm(tangent2, axis=1, keepdims=True)

        return normals, tangent1, tangent2

    def calculate_points(self, resolution):
        x = np.linspace(self.Xfine.min(), self.Xfine.max(), resolution)
        y = np.linspace(self.Yfine.min(), self.Yfine.max(), resolution)

        X, Y = np.meshgrid(x, y)
        Z = self.__evaluate_at_points__(x, y)
        test_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        normals, tangent1, tangent2 = self.__calculate_normals_tangents__(x, y)

        # Check orthogonality
        dot_12 = np.einsum('ij,ij->i', tangent1, tangent2)
        dot_1n = np.einsum('ij,ij->i', tangent1, normals)
        dot_2n = np.einsum('ij,ij->i', tangent2, normals)
        print("Max deviation from orthogonality:", np.max(np.abs([dot_12, dot_1n, dot_2n])))

        return test_points, normals, tangent1, tangent2

    def calculate_dataframe(self, resolution):
        points, normals, t1, t2 = self.calculate_points(resolution)
        df = pd.DataFrame({
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2],
            'normal_x': normals[:, 0],
            'normal_y': normals[:, 1],
            'normal_z': normals[:, 2],
            'tau1_x': t1[:, 0],
            'tau1_y': t1[:, 1],
            'tau1_z': t1[:, 2],
            'tau2_x': t2[:, 0],
            'tau2_y': t2[:, 1],
            'tau2_z': t2[:, 2],
        })
        return df

# from spline_surface import Spline

# # Assume Xfine, Yfine, Zfine are 2D numpy arrays from a meshgrid
# spline = Spline(Xfine, Yfine, Zfine)

# df = spline.calculate_dataframe(resolution=50)
# df.to_csv("surface_with_frames.csv", index=False)
