#include <iostream>
#include <Eigen/Dense>
#include "SurfaceSphere.h"
#include "Constants.h"

int mainSurface() {
    double radius = 1.0;
    Eigen::Vector3d center(0.0, 0.0, 1.0);
    int resolution = 10;

    SurfaceSphere surface(radius, center, resolution); // ✅ Fixed

    const auto& points = surface.getPoints();
    const auto& normals = surface.getNormals();
    const auto& tau1 = surface.getTau1();
    const auto& tau2 = surface.getTau2();

    std::cout << "SphereSurface with " << points.rows() << " points generated.\n";
    std::cout << "First point: " << points.row(0) << "\n";
    std::cout << "Normal at first point: " << normals.row(0) << "\n";
    std::cout << "Tau1 at first point: " << tau1.row(0) << "\n";
    std::cout << "Tau2 at first point: " << tau2.row(0) << "\n";

    auto mirrored = surface.mirrored(Eigen::Vector3d(0, 0, 1));
    std::cout << "Mirrored first point: " << mirrored->getPoints().row(0) << "\n";

    return 0;
}
