import numpy as np
from scipy.interpolate import bisplev, bisplrep

class Spline:
    def __init__(self, Xfine, Yfine, Zfine):
        self.Xfine = Xfine
        self.Yfine = Yfine
        self.Zfine = Zfine
        
        self.tcks = bisplrep(Xfine.ravel(), Yfine.ravel(), Zfine.ravel(), s=0.5)
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
        x = np.linspace(self.Xfine.min(), self.Xfine.max(), resolution)
        y = np.linspace(self.Yfine.min(), self.Yfine.max(), resolution)

        X, Y = np.meshgrid(x, y)
        Z = self.__evaluate_at_points__(x, y)
        test_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        normals, tangent1, tangent2 = self.__calculate_normals_tangents__(x, y)


        # if control_points_flag:
        #     control_x = np.linspace(self.Xfine.min() + (x[1]-x[0])/2, self.Xfine.max()- (x[1]-x[0])/2, resolution - 1)
        #     control_y = np.linspace(self.Yfine.min() + (y[1]-y[0])/2, self.Yfine.max()- (y[1]-y[0])/2, resolution - 1)
            
        #     X_control, Y_control = np.meshgrid(control_y, control_y)
        #     Z_control  = self.__evaluate_at_points__(control_x, control_y)
        #     control_points = np.column_stack((X_control.ravel(), Y_control.ravel(), Z_control.ravel()))

        #     _, control_tangent1, control_tangent2 = self.__calculate_normals_tangents__(control_x, control_y)

        #     return test_points, normals, tangent1, tangent2, control_points, control_tangent1, control_tangent2

        dot_12 = np.einsum('ij,ij->i', tangent1, tangent2)
        dot_1n = np.einsum('ij,ij->i', tangent1, normals)
        dot_2n = np.einsum('ij,ij->i', tangent2, normals)

        print("Max deviation from orthogonality:", np.max(np.abs([dot_12, dot_1n, dot_2n])))


    
        return test_points, normals, tangent1, tangent2
