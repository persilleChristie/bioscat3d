#ifndef FIELDCALCULATOR_H
#define FIELDCALCULATOR_H

#include <Eigen/Dense>

/// @brief Base class for field calculators.
/// @details This class defines the
/// interface for computing electromagnetic fields at specified evaluation points.
class FieldCalculator {
public:

    /// @brief Compute the electromagnetic fields at specified evaluation points.
    /// @param outE Output matrix for electric field components (Nx3 complex numbers).
    /// @param outH Output matrix for magnetic field components (Nx3 complex numbers).
    /// @param evalPoints Matrix of evaluation points (Nx3 real numbers).
    /// @param polarization_idx Index of the polarization to use (default is 0, used in total fields).
    virtual void computeFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints, 
        int polarization_idx = 0 // Only used in total fields
    ) const = 0;

    virtual void computeFarFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints, 
        int polarization_idx = 0 // Only used in total fields
    ) const = 0;

    virtual Eigen::VectorXd computePower(
        const Surface& surface
    ) const = 0;

    /// @brief Compute the tangential components of the electric and magnetic fields.
    /// @param outE Output matrix for electric field components (Nx3 complex numbers).
    /// @param outH Output matrix for magnetic field components (Nx3 complex numbers).
    /// @param outEt1 Output matrix for tangential electric field component along tau1 (Nx3 complex numbers).
    /// @param outEt2 Output matrix for tangential electric field component along tau2 (Nx3 complex numbers).
    /// @param outHt1 Output matrix for tangential magnetic field component along tau1 (Nx3 complex numbers).
    /// @param outHt2 Output matrix for tangential magnetic field component along tau2 (Nx3 complex numbers).
    /// @param tau1 Tangential vector 1 (Nx3 real numbers).
    /// @param tau2 Tangential vector 2 (Nx3 real numbers).
    virtual void computeTangentialFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        Eigen::MatrixX3cd& outEt1,
        Eigen::MatrixX3cd& outEt2,
        Eigen::MatrixX3cd& outHt1,
        Eigen::MatrixX3cd& outHt2,
        const Eigen::MatrixX3d& tau1,
        const Eigen::MatrixX3d& tau2
    ) const {

        int N = outE.rows();

        for (int i = 0; i < N; ++i) {
            outEt1.row(i) = outE.row(i).dot(tau1.row(i))*tau1.row(i);
            outEt2.row(i) = outE.row(i).dot(tau2.row(i))*tau2.row(i);
            outHt1.row(i) = outH.row(i).dot(tau1.row(i))*tau1.row(i);
            outHt2.row(i) = outH.row(i).dot(tau2.row(i))*tau2.row(i);
        }
    }

    virtual ~FieldCalculator() = default;
};

#endif // FIELDCALCULATOR_H
