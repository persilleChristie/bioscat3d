import numpy as np
from scipy.interpolate import bisplev, bisplrep

class Spline:
    def __init__(self, Xfine, Yfine, Zfine):
        self.Xfine = Xfine
        self.Yfine = Yfine
        self.Zfine = Zfine
        
        self.tcks = bisplrep(Xfine.ravel(), Yfine.ravel(), Zfine.ravel())
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

        denom = (1 + fxx**2 + fyy**2)**1.5
        numer = (1 + fx**2) * fyy - 2 * fx * fy * fxy + (1 + fy**2) * fxx

        H = numer / (2 * denom)

        return np.max(abs(H))
    
    def calculate_points(self, resolution):
        x = np.linspace(self.Xfine.min(), self.Xfine.max(), resolution)
        y = np.linspace(self.Yfine.min(), self.Yfine.max(), resolution)

        x_control = np.linspace(self.Xfine.min() + (x[1]-x[0])/2, self.Xfine.max()- (x[1]-x[0])/2, resolution - 1)
        y_control = np.linspace(self.Yfine.min() + (y[1]-y[0])/2, self.Yfine.max()- (y[1]-y[0])/2, resolution - 1)

        X, Y = np.meshgrid(x, y)
        Z = self.__evaluate_at_points__(x, y)

        X_control, Y_control = np.meshgrid(x_control, y_control)
        Z_control  = self.__evaluate_at_points__(x_control, y_control)

        test_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        control_points = np.column_stack((X_control.ravel(), Y_control.ravel(), Z_control.ravel()))

        # Compute normals
        fx = self.__evaluate_at_points__(x, y, dx = 1).ravel()
        fy = self.__evaluate_at_points__(x, y, dy = 1).ravel()
    
        normals = np.column_stack((-fx, -fy, np.ones_like(fx)))

        # Pick reference vector based on condition per row
        x_axis = np.array([1, 0, 0])
        z_axis = np.array([0, 0, 1])
        ref = np.where(np.abs(normals[:, 2:3]) < 0.9, z_axis, x_axis)  # shape: (N, 3)

        # Project ref onto the normal and subtract for orthogonal component
        dot_ref_n = np.einsum('ij,ij->i', ref, normals)[:, np.newaxis]  # (N, 1)
        tangent1 = ref - dot_ref_n * normals
        tangent1 /= np.linalg.norm(tangent1, axis=1, keepdims=True)

        # Compute second tangent as cross product
        tangent2 = np.cross(normals, tangent1)

        return test_points, normals, tangent1, tangent2, control_points



# class SplineManager:
#     _spline_instance = None

#     @staticmethod
#     def initialize(Xfine, Yfine, Zfine):
#         if SplineManager._spline_instance is None:
#             SplineManager._spline_instance = Spline(Xfine, Yfine, Zfine)
#         else:
#             # Update data without rebuilding object
#             spline = SplineManager._spline_instance
#             spline.Xfine = Xfine
#             spline.Yfine = Yfine
#             spline.Zfine = Zfine
#             spline.tcks = bisplrep(Xfine.ravel(), Yfine.ravel(), Zfine.ravel())
#             spline.max_curvature = spline.__compute_max_mean_curvature__()

#     @staticmethod
#     def get_max_curvature():
#         return SplineManager._spline_instance.max_curvature

#     @staticmethod
#     def calculate(resolution):
#         return SplineManager._spline_instance.calculate_points(resolution) 
    
    

