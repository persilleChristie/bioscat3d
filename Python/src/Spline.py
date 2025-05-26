import numpy as np
from scipy.interpolate import bisplev, bisplrep

class Spline:
    def __init__(self, Xfine, Yfine, Zfine, S = 0.5):
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

        #denom = (1 + fxx**2 + fyy**2)**1.5 # fejl her
        numer = (1 + fx**2) * fyy - 2 * fx * fy * fxy + (1 + fy**2) * fxx
        denom = (1 + fx**2 + fy**2)**1.5

        H = numer / (2 * denom)

        return np.max(abs(H))
    
    def __calculate_normals_tangents__(self, x, y):
        # X, Y = np.meshgrid(x, y)
        # Z = self.__evaluate_at_points__(x, y)

        # points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # Compute normals
        fx = self.__evaluate_at_points__(x, y, dx = 1).ravel()
        fy = self.__evaluate_at_points__(x, y, dy = 1).ravel()
    
        # normals = np.column_stack((-fx, -fy, np.ones_like(fx)))

        # # Pick reference vector based on condition per row
        # x_axis = np.array([1, 0, 0])
        # z_axis = np.array([0, 0, 1])
        # ref = np.where(np.abs(normals[:, 2:3]) < 0.9, z_axis, x_axis)  # shape: (N, 3)

        # # Project ref onto the normal and subtract for orthogonal component
        # dot_ref_n = np.einsum('ij,ij->i', ref, normals)[:, np.newaxis]  # (N, 1)
        # tangent1 = ref - dot_ref_n * normals
        # tangent1 /= np.linalg.norm(tangent1, axis=1, keepdims=True)

        # # Compute second tangent as cross product
        # tangent2 = np.cross(normals, tangent1)

        tangent1 = np.stack([np.ones_like(fx), np.zeros_like(fx), fx], axis=-1)
        tangent2 = np.stack([np.zeros_like(fy), np.ones_like(fy), fy], axis=-1)

        normals = np.cross(tangent1, tangent2)
        tangent2 = np.cross(normals, tangent1)  # Ensure right-handed orthonormal basis

        # Normalize
        tangent1 /= np.linalg.norm(tangent1, axis=-1, keepdims=True)
        tangent2 /= np.linalg.norm(tangent2, axis=-1, keepdims=True)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        return normals, tangent1, tangent2
    
    def calculate_points(self, resolution):
        # x = np.linspace(self.Xfine.min(), self.Xfine.max(), resolution)
        # y = np.linspace(self.Yfine.min(), self.Yfine.max(), resolution)

        # X, Y = np.meshgrid(x, y)
        # Z = self.__evaluate_at_points__(x, y)
        # test_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # normals, tangent1, tangent2 = self.__calculate_normals_tangents__(x, y)

        # dot_12 = np.einsum('ij,ij->i', tangent1, tangent2)
        # dot_1n = np.einsum('ij,ij->i', tangent1, normals)
        # dot_2n = np.einsum('ij,ij->i', tangent2, normals)

        # print("Max deviation from orthogonality:", np.max(np.abs([dot_12, dot_1n, dot_2n])))
        # print("Minimum value of normal z-value: ", np.min(normals[:,2]))


    
        # return test_points, normals, tangent1, tangent2
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
