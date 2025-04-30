import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

def generate_grid(halfWidth=1, resol=20):
    x = np.linspace(-halfWidth, halfWidth, resol)
    y = np.linspace(-halfWidth, halfWidth, resol)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    return x, y, X, Y, Z

def build_surface_from_bumps(X, Y, bump_parameters):
    """
    Build a surface Z by adding multiple Gaussian bumps
    Parameters:
        X, Y: meshgrid arrays
        bump_parameters: numpy array of shape (N, 4) where each row is (x0, y0, height, sigma)
    Returns:
        Z: the resulting surface (same shape as X and Y)
    """
    Z = np.zeros_like(X)
    for bump in bump_parameters:
        x0, y0, height, sigma = bump
        Z += height * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    return Z

def get_MAS_surface_data(data):
    """
    Assemble MAS surface data (test points, normals, tangents, auxiliary points) into structured arrays.
    Returns:
        test_points: (N, 3) array
        test_normals: (N, 3) array
        tangent1: (N, 3) array
        tangent2: (N, 3) array
        aux_points_int: (M, 3) array
        aux_points_ext: (M, 3) array
        aux_indices: (M,) array (index of test point corresponding to each aux point pair)
    """
    test_points = data["test_points"]
    test_normals = data["test_normals"]
    tangent1 = data["tangent1"]
    tangent2 = data["tangent2"]
    aux_points_int = data["aux_points_int"]
    aux_points_ext = data["aux_points_ext"]
    aux_indices = data["aux_indices"]

    return test_points, test_normals, tangent1, tangent2, aux_points_int, aux_points_ext, aux_indices

def approx_peak_curvature(Z, x, y):
    peak_idx = np.unravel_index(np.argmax(Z), Z.shape)
    dz_dx2 = np.gradient(np.gradient(Z, x, axis=1), x, axis=1)
    dz_dy2 = np.gradient(np.gradient(Z, y, axis=0), y, axis=0)
    curv_xx = dz_dx2[peak_idx]
    curv_yy = dz_dy2[peak_idx]
    mean_curv = 0.5 * (curv_xx + curv_yy)
    return mean_curv

def generate_MAS_model(x, y, X, Y, Z, lambda_0=0.325, d = 0.14, testPointPerLambda = 5):
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]
    N_test = int(np.ceil(2 * (testPointPerLambda * Lx * testPointPerLambda * Ly) / lambda_0**2))

    # --- Step 3: Flatten grid and remove edge points ---
    X_flat, Y_flat, Z_flat = X.ravel(), Y.ravel(), Z.ravel()
    points = np.stack([X_flat, Y_flat, Z_flat], axis=1)
    mask = (X_flat > x[0]) & (X_flat < x[-1]) & (Y_flat > y[0]) & (Y_flat < y[-1])
    interior_points = points[mask]

    total_pts = interior_points.shape[0]

    # --- Step 4: Sample test points ---
    idx_test = np.arange(0, total_pts, 2)  # ✅ every second point in order
    test_points = interior_points[idx_test] # ✅ preserves structure

    # After selecting test_points
    N_test = len(test_points)  # ✅ actual number used

    # For aux indices
    aux_test_indices = np.arange(0, N_test, 2)  # ✅ safe indexing

    # --- Step 5: Compute normals and tangents ---
    dz_dx, dz_dy = np.gradient(Z, x, y)
    dz_dx_flat = dz_dx.ravel()[mask]
    dz_dy_flat = dz_dy.ravel()[mask]

    normals = np.stack([-dz_dx_flat, -dz_dy_flat, np.ones_like(dz_dx_flat)], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    test_normals = normals[idx_test]

    # Compute tangent vectors (Gram-Schmidt)
    tangent1 = np.zeros_like(test_normals)
    tangent2 = np.zeros_like(test_normals)

    for i, n in enumerate(test_normals):
        if abs(n[2]) < 0.9:
            ref = np.array([0, 0, 1])
        else:
            ref = np.array([1, 0, 0])

        t1 = ref - np.dot(ref, n) * n
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        tangent1[i] = t1
        tangent2[i] = t2

    # --- Step 6: Compute auxiliary points and mapping indices ---
    aux_test_indices = np.arange(N_test)[::2]
    aux_test_points = test_points[aux_test_indices]
    aux_test_normals = test_normals[aux_test_indices]

    aux_points_int = aux_test_points - d * aux_test_normals
    aux_points_ext = aux_test_points + d * aux_test_normals

    # --- Return full structured data ---
    return {
    "test_points": test_points,
    "test_normals": test_normals,
    "tangent1": tangent1,
    "tangent2": tangent2,
    "aux_points_int": aux_points_int,
    "aux_points_ext": aux_points_ext,
    "aux_indices": aux_test_indices,
    }

def visualize_surface_and_points(X, Y, Z, mas_data, path_prefix):
    def plot_points(points, color, name, filename):
        fig = go.Figure()
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.6))
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                   mode='markers', marker=dict(size=2, color=color), name=name))
        fig.update_layout(scene=dict(xaxis_title='x (μm)', yaxis_title='y (μm)', zaxis_title='z (μm)', aspectmode='data'),
                          title=f"3D Surface with {name}")
        fig.write_html(filename)
        print(f"✅ Saved: {filename}")

    # Plot all combined
    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.6))
    fig.add_trace(go.Scatter3d(x=mas_data["test_points"][:, 0], y=mas_data["test_points"][:, 1], z=mas_data["test_points"][:, 2],
                               mode='markers', marker=dict(size=2, color='red'), name="Test points"))
    fig.add_trace(go.Scatter3d(x=mas_data["aux_points_int"][:, 0], y=mas_data["aux_points_int"][:, 1], z=mas_data["aux_points_int"][:, 2],
                               mode='markers', marker=dict(size=2, color='orange'), name="Aux (int)"))
    fig.add_trace(go.Scatter3d(x=mas_data["aux_points_ext"][:, 0], y=mas_data["aux_points_ext"][:, 1], z=mas_data["aux_points_ext"][:, 2],
                               mode='markers', marker=dict(size=2, color='cyan'), name="Aux (ext)"))
    fig.update_layout(scene=dict(xaxis_title='x (μm)', yaxis_title='y (μm)', zaxis_title='z (μm)', aspectmode='data'),
                      title="Interactive 3D Surface with Points")
    fig.write_html(path_prefix + "/surface_with_points.html")

    # Individual sets
    plot_points(mas_data["test_points"], 'red', "Test Points", path_prefix + "/surface_test_points.html")
    plot_points(mas_data["aux_points_int"], 'orange', "Auxiliary Interior Points", path_prefix + "/surface_aux_int.html")
    plot_points(mas_data["aux_points_ext"], 'cyan', "Auxiliary Exterior Points", path_prefix + "/surface_aux_ext.html")

    # Test points with and without aux
    mask_with_aux = np.zeros(len(mas_data["test_points"]), dtype=bool)
    mask_with_aux[mas_data["aux_indices"]] = True
    test_with_aux = mas_data["test_points"][mask_with_aux]
    test_without_aux = mas_data["test_points"][~mask_with_aux]

    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.6))
    fig.add_trace(go.Scatter3d(x=test_with_aux[:, 0], y=test_with_aux[:, 1], z=test_with_aux[:, 2],
                               mode='markers', marker=dict(size=3, color='blue'), name="Test points with aux"))
    fig.add_trace(go.Scatter3d(x=test_without_aux[:, 0], y=test_without_aux[:, 1], z=test_without_aux[:, 2],
                               mode='markers', marker=dict(size=2, color='gray'), name="Other test points"))
    fig.update_layout(scene=dict(xaxis_title='x (μm)', yaxis_title='y (μm)', zaxis_title='z (μm)', aspectmode='data'),
                      title="Test Points With vs Without Aux Points")
    fig.write_html(path_prefix + "/test_points_with_vs_without_aux.html")
    print("✅ Saved: ./test_points_with_vs_without_aux.html")



def main():
    data_path = "./"
    halfWidth = 1
    resol = 20
    x, y, X, Y, Z = generate_grid(halfWidth, resol)
    
    # --- Generate random bump parameters first ---
    num_bumps = 3
    bump_parameters = np.zeros((num_bumps, 4))

    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)

    np.random.seed(42)
    for i in range(num_bumps):
        x0 = np.random.uniform(x_min + 0.1*(x_max - x_min), x_max - 0.1*(x_max - x_min))
        y0 = np.random.uniform(y_min + 0.1*(y_max - y_min), y_max - 0.1*(y_max - y_min))
        height = np.random.uniform(0.1, 0.4)
        sigma = np.random.uniform(0.2, 0.7)
        bump_parameters[i] = [x0, y0, height, sigma]

    # --- Build surface from bump parameters ---
    Z = build_surface_from_bumps(X, Y, bump_parameters)

    mean_curv = approx_peak_curvature(Z, x, y)
    radius = 1 / abs(mean_curv)
    d = 0.14 * radius
    print(f"Final offset d = {d:.6f} μm")

    data = generate_MAS_model(x, y, X, Y, Z, d = d)

    # Access results
    test_pts = data["test_points"]
    normals = data["test_normals"]
    t1 = data["tangent1"]
    t2 = data["tangent2"]
    aux_int = data["aux_points_int"]
    aux_ext = data["aux_points_ext"]
    aux_test_idx = data["aux_indices"]

    visualize_surface_and_points(X, Y, Z, data, data_path)

if __name__ == "__main__":
    main()
