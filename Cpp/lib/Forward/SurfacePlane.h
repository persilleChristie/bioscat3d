#ifndef PLANE_SURFACE_H
#define PLANE_SURFACE_H

#include "Surface.h"
#include "../Utils/Constants.h"
#include <Eigen/Dense>

/// @brief Class representing a plane surface in the Forward module.
/// This class inherits from the Surface class and provides methods to generate and access
/// points, normals, and tangent vectors on a plane defined by a corner point, two basis vectors,
/// and sizes along those basis vectors.
class SurfacePlane : public Surface {
public:
    /// @brief Constructor for the SurfacePlane class.
    /// @param cornerPoint The corner point of the plane.
    /// @param basis1 The first basis vector defining the plane.
    /// @param basis2 The second basis vector defining the plane.
    /// @param size1 The size along the first basis vector.
    /// @param size2 The size along the second basis vector.
    /// @param resolution The resolution of the surface grid.
    SurfacePlane(Eigen::Vector3d cornerPoint,
        Eigen::Vector3d basis1,
        Eigen::Vector3d basis2,
        double size1,
        double size2,
        int resolution);
    /// @brief Destructor for the SurfacePlane class.
    ~SurfacePlane() override = default;

    /// @brief Returns the points on the surface.
    /// @return A constant reference to the matrix of points on the surface.
    const Eigen::MatrixXd& getPoints() const override;

    /// @brief Returns the normals at the points on the surface.
    /// @return A constant reference to the matrix of normals at the surface points.
    const Eigen::MatrixXd& getNormals() const override;

    /// @brief Returns the first tangent vector (tau1) at the surface points.
    /// @return A constant reference to the matrix of first tangent vectors at the surface points.
    const Eigen::MatrixXd& getTau1() const override;
    
    /// @brief Returns the second tangent vector (tau2) at the surface points.
    /// @return A constant reference to the matrix of second tangent vectors at the surface points.
    const Eigen::MatrixXd& getTau2() const override;


private:   
    Eigen::Vector3d cornerPoint_;
    Eigen::Vector3d basis1_;
    Eigen::Vector3d basis2_;
    double size1_;
    double size2_;
    int resolution_;

    Eigen::MatrixXd points_;
    Eigen::MatrixXd normals_;
    Eigen::MatrixXd tau1_;
    Eigen::MatrixXd tau2_;

    /// @brief Generates the surface points, normals, and tangent vectors.
    /// This method computes the grid of points on the plane, calculates the normals,
    /// and computes the tangent vectors tau1 and tau2 based on the basis vectors and resolution.
    /// It is called during the construction of the SurfacePlane object.
    void generateSurface();
};
#endif // PLANE_SURFACE_H

