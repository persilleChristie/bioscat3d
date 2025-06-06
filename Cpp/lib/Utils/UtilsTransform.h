#ifndef UTILS_TRANSFORM_H
#define UTILS_TRANSFORM_H


#include <Eigen/Dense>
#include <cmath>

/// @brief Utility functions for coordinate transformations and E-field computations.
/// @file UtilsTransform.h
/// @namespace TransformUtils
/// @details This namespace provides functions to compute angles, rotation matrices,
///          and E-field components in Cartesian coordinates based on spherical coordinates.
namespace TransformUtils {

    /// @brief Computes the angles theta and phi from a 3D vector and its magnitude.
    /// @param x The 3D vector.
    /// @param r The magnitude of the vector.
    /// @param cosTheta Output cosine of theta.
    /// @param sinTheta Output sine of theta.
    /// @param cosPhi Output cosine of phi.
    /// @param sinPhi Output sine of phi.
    /// @details If r is zero, sets angles to zero and cosPhi to 1.
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

    /// @brief Calculates the rotation matrix for a rotation around the Y-axis.
    /// @param cosTheta Cosine of the angle of rotation.
    /// @param sinTheta Sine of the angle of rotation.
    /// @return The rotation matrix for the Y-axis.
    inline Eigen::Matrix3d rotationMatrixY(double cosTheta, double sinTheta) {
        return (Eigen::Matrix3d() <<
            cosTheta, 0, sinTheta,
            0,        1, 0,
            -sinTheta, 0, cosTheta).finished();
    }

    /// @brief Calculates the rotation matrix for a rotation around the Z-axis.
    /// @param cosPhi Cosine of the angle of rotation.
    /// @param sinPhi Sine of the angle of rotation.
    /// @return The rotation matrix for the Z-axis.
    inline Eigen::Matrix3d rotationMatrixZ(double cosPhi, double sinPhi) {
        return (Eigen::Matrix3d() <<
            cosPhi, -sinPhi, 0,
            sinPhi,  cosPhi, 0,
            0,       0,      1).finished();
    }

    /// @brief Calculates the inverse rotation matrix for a rotation around the Y-axis.
    /// @param cosTheta Cosine of the angle of rotation.
    /// @param sinTheta Sine of the angle of rotation.
    /// @return The inverse rotation matrix for the Y-axis.
    /// @details This is the transpose of the rotation matrix for Y-axis.
    inline Eigen::Matrix3d rotationMatrixYInv(double cosTheta, double sinTheta) {
        return (Eigen::Matrix3d() <<
            cosTheta, 0, -sinTheta,
            0,        1, 0,
            sinTheta, 0, cosTheta).finished();
    }

    /// @brief Calculates the inverse rotation matrix for a rotation around the Z-axis.
    /// @param cosPhi Cosine of the angle of rotation.
    /// @param sinPhi Sine of the angle of rotation.
    /// @return The inverse rotation matrix for the Z-axis.
    /// @details This is the transpose of the rotation matrix for Z-axis.
    inline Eigen::Matrix3d rotationMatrixZInv(double cosPhi, double sinPhi) {
        return (Eigen::Matrix3d() <<
            cosPhi, sinPhi, 0,
            -sinPhi,  cosPhi, 0,
            0,       0,      1).finished();
    }

    /// @brief Translates a vector by subtracting a point.
    /// @param v The vector to translate.
    /// @param xPrime The point to subtract from the vector.
    /// @return The translated vector.
    /// @details This function computes v - xPrime.
    ///          It is useful for transforming vectors relative to a specific point.
    inline Eigen::Vector3d translateVector(const Eigen::Vector3d& v, const Eigen::Vector3d& xPrime) {
        return v - xPrime;
    }

    // =========================================
    //  Compute E-Field Components
    // =========================================
    /// @brief Computes the E-field components in Cartesian coordinates from spherical coordinates.
    /// @param sin_theta Sine of the polar angle theta.
    /// @param cos_theta Cosine of the polar angle theta.
    /// @param sin_phi Sine of the azimuthal angle phi.
    /// @param cos_phi Cosine of the azimuthal angle phi.
    /// @param E_r The radial component of the E-field.
    /// @param E_theta The polar component of the E-field.
    /// @return The E-field components in Cartesian coordinates as a 3D vector.
    /// @details The E-field components are computed as follows:
    ///          E_x = E_r * sin(theta) * cos(phi) + E_theta * cos(theta) * cos(phi)
    ///          E_y = E_r * sin(theta) * sin(phi) + E_theta * cos(theta) * sin(phi)
    ///          E_z = E_r * cos(theta) - E_theta * sin(theta)
    ///         - θ (theta): polar angle from +Z
    ///         - φ (phi): azimuthal angle from +X in the XY plane
    /// @note This function assumes that the input angles are in radians.
    inline Eigen::Vector3cd computeECartesian(double sin_theta, double cos_theta, 
        double sin_phi, double cos_phi, 
        const std::complex<double>& E_r, 
        const std::complex<double>& E_theta) {
    return {
    E_r * sin_theta * cos_phi + E_theta * cos_theta * cos_phi,
    E_r * sin_theta * sin_phi + E_theta * cos_theta * sin_phi,
    E_r * cos_theta - E_theta * sin_theta
    };
    }

} // namespace TransformUtils

#endif // UTILS_TRANSFORM_H
