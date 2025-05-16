#include "SurfaceFromGP.h"
#include <unsupported/Eigen/MatrixFunctions> // for matrix operations if needed
#include <stdexcept>

SurfaceFromGP::SurfaceFromGP(const Eigen::MatrixXd& points)
    : points_(points) {
    if (points_.cols() != 3) {
        throw std::invalid_argument("SurfaceFromGP expects 3D points (n x 3)");
    }
    computeTangentsAndNormals();
}

std::unique_ptr<Surface> SurfaceFromGP::mirrored(const Eigen::Vector3d& normal) const {
    Eigen::MatrixXd mirrored_points = points_;
    for (int i = 0; i < mirrored_points.rows(); ++i) {
        Eigen::Vector3d p = mirrored_points.row(i);
        mirrored_points.row(i) = p - 2.0 * (p.dot(normal)) * normal;
    }
    return std::make_unique<SurfaceFromGP>(mirrored_points);
}

void SurfaceFromGP::computeTangentsAndNormals() {
    const int n = points_.rows();
    tau1_.resize(n, 3);
    tau2_.resize(n, 3);
    normals_.resize(n, 3);

    // Simple finite difference-based estimation for small neighborhood structure
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d p = points_.row(i);

        // Use fixed epsilon shift to build local tangents (approximate)
        Eigen::Vector3d dx(1e-2, 0.0, 0.0);
        Eigen::Vector3d dy(0.0, 1e-2, 0.0);

        Eigen::Vector3d px = p + dx;
        Eigen::Vector3d py = p + dy;

        // Project these back to surface z(x, y) = f(x,y) (approximate tangent planes)
        px.z() = p.z();  // we assume local flatness for now
        py.z() = p.z();

        Eigen::Vector3d t1 = (px - p).normalized();
        Eigen::Vector3d t2 = (py - p).normalized();
        Eigen::Vector3d n = t1.cross(t2).normalized();

        tau1_.row(i) = t1;
        tau2_.row(i) = t2;
        normals_.row(i) = n;
    }
}
