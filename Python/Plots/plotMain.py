import numpy as np
import plotly.graph_objects as go
import json

def gaussian_bump(x, y, x0, y0, height, sigma):
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return height * np.exp(-r2 / (2 * sigma ** 2))

def compute_surface(json_data):
    # Extract geometry
    halfWidth_x = json_data["halfWidth_x"]
    halfWidth_y = json_data["halfWidth_y"]
    resolution = json_data["resolution"]
    bumps = json_data["bumpData"]

    # Define grid
    N = 2 * resolution + 1
    x_vals = np.linspace(-halfWidth_x, halfWidth_x, N)
    y_vals = np.linspace(-halfWidth_y, halfWidth_y, N)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    # Sum bumps
    for bump in bumps:
        Z += gaussian_bump(X, Y, bump["x0"], bump["y0"], bump["height"], bump["sigma"])

    return X, Y, Z

def plot_surface(X, Y, Z, output_html="surface_from_json.html"):
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    fig.update_layout(title="Bumpy Surface from JSON",
                      margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    fig.write_html(output_html)
    print(f"[✓] Plot saved to: {output_html}")

# -------------------------------
# Main: Load JSON & Plot Surface
# -------------------------------

# Paste JSON into a string or load from file
with open("../../json/surfaceParamsPseudoReal05.json", "r") as f:
    json_data = json.load(f)


X, Y, Z = compute_surface(json_data)
plot_surface(X, Y, Z)
