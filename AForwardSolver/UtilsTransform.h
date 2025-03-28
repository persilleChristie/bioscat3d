#ifndef UTILSTRANSFORM_H
#define UTILSTRANSFORM_H

#include <Eigen/Dense>
#include <cmath>

namespace TransformUtils {

inline void computeAngles(const Eigen::Vector3d& x, double r,
                          double& cosTheta, double& sinTheta,
                          double& cosPhi, double& sinPhi) {
    if (r == 0.0) {
        cosTheta = sinTheta = 0.0;
        cosPhi = 1.0; sinPhi = 0.0;
        return;
    }

    cosTheta = x[2] / r;
    sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
    double xy_norm = std::hypot(x[0], x[1]);

    if (xy_norm == 0.0) {
        cosPhi = 1.0;
        sinPhi = 0.0;
    } else {
        cosPhi = x[0] / xy_norm;
        sinPhi = x[1] / xy_norm;
    }
}

inline Eigen::Matrix3d rotationMatrixY(double cosTheta, double sinTheta) {
    return (Eigen::Matrix3d() <<
        cosTheta, 0, sinTheta,
        0,        1, 0,
        -sinTheta, 0, cosTheta).finished();
}

inline Eigen::Matrix3d rotationMatrixZ(double cosPhi, double sinPhi) {
    return (Eigen::Matrix3d() <<
        cosPhi, -sinPhi, 0,
        sinPhi,  cosPhi, 0,
        0,       0,      1).finished();
}

inline Eigen::Vector3d translateVector(const Eigen::Vector3d& v, const Eigen::Vector3d& xPrime) {
    return v - xPrime;
}

} // namespace TransformUtils

#endif // TRANSFORMUTILS_H