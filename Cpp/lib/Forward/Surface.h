#ifndef SURFACE_H
#define SURFACE_H

#include <Eigen/Dense>
#include <memory>

/// @brief Abstract base class for surfaces in the Forward module.
/// This class defines the interface for surfaces, providing methods to access points,
/// normals, and tangent vectors (tau1 and tau2).
class Surface {
public:
    /// @brief Default constructor for the Surface class.
    virtual ~Surface() = default;

    virtual const Eigen::MatrixXd& getPoints() const = 0;
    virtual const Eigen::MatrixXd& getNormals() const = 0;
    virtual const Eigen::MatrixXd& getTau1() const = 0;
    virtual const Eigen::MatrixXd& getTau2() const = 0;
};

#endif // SURFACE_H
