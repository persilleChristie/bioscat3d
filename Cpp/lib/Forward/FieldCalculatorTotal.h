#ifndef FIELDCALCULATORTOTAL_H
#define FIELDCALCULATORTOTAL_H

#include "FieldCalculator.h"
#include "../Utils/Constants.h"
#include "MASSystem.h"
#include <Eigen/Dense>
#include "Surface.h"

/// @brief FieldCalculator for computing total fields from multiple sources, including dipoles and plane waves.
/// @details This class extends the FieldCalculator class to implement the computation of
/// total electric and magnetic fields at specified evaluation points. It combines contributions from
/// multiple dipoles and plane waves, allowing for the calculation of fields in both interior and exterior regions.
/// It also provides methods for computing power and tangential error of the fields.
class FieldCalculatorTotal : public FieldCalculator {
public:
    /// @brief Constructor for FieldCalculatorTotal.
    /// @param masSystem The MASSystem object containing the surface and incident field parameters.
    /// @param verbose Boolean indicating whether to print verbose output during initialization (default is false).
    /// @details This constructor initializes the total field calculator with the provided MASSystem,
    /// setting up the necessary dipoles and plane waves based on the system's parameters.
    FieldCalculatorTotal(const MASSystem& masSystem, bool verbose = false);


    /// @brief Computes the total electric and magnetic fields at specified evaluation points.
    /// @details This method implements the computation of the total electric and magnetic fields
    /// generated by the dipoles and plane waves at the given evaluation points. The fields are returned in the output matrices `outE` and `outH`.
    /// @param outE Output matrix for electric field components (Nx3 complex numbers).
    /// @param outH Output matrix for magnetic field components (Nx3 complex numbers).
    /// @param evalPoints Matrix of evaluation points (Nx3 real numbers).
    /// @param polarization_idx Index of the polarization to use (default is 0, used in total fields).
    void computeFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints,
        int polarization_idx = 0
    ) const override;

    void computeFarFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints,
        int polarization_idx = 0
    ) const override;

    /// @brief Computes the total power radiated by the surface.
    /// @details This method computes the total power radiated by the surface based on the electric and magnetic fields.
    /// It integrates the Poynting vector over the surface area defined by the evaluation points and normals.
    /// @param surface The Surface object containing the evaluation points and normals.
    /// @return A vector containing the total power for each polarization. 
    Eigen::VectorXd computePower(
        const Surface& surface
    ) const override;

    /// @brief Gets the amplitudes of the fields.
    /// @details This method returns the amplitudes of the fields as a matrix, where each row corresponds to a polarization
    /// and each column corresponds to an evaluation point.
    /// @return A matrix of complex amplitudes (BxN, where B is the number of polarizations and N is the number of evaluation points).
    [[nodiscard]] const Eigen::MatrixXcd& getAmplitudes() const { return amplitudes_; }

    /// @brief Computes the tangential error of the fields.
    /// @details This method computes the tangential components of the electric and magnetic fields at the control points
    /// and calculates the error between the tangential components of the fields and the control tangents.
    /// It returns the tangential errors for both electric and magnetic fields as vectors.
    /// @param polarization_index Index of the polarization to use for the tangential error calculation.
    /// @return A vector containing the tangential errors for electric and magnetic fields.
    /// The first half of the vector contains the electric field errors, and the second half contains the magnetic field errors.
    std::vector<Eigen::VectorXd> computeTangentialError(int polarization_index);

private:
    Eigen::MatrixXcd amplitudes_;
    Eigen::MatrixXcd amplitudes_ext_;
    std::vector<std::shared_ptr<FieldCalculator>> dipoles_;
    std::vector<std::shared_ptr<FieldCalculator>> dipoles_ext_;
    std::vector<std::shared_ptr<FieldCalculator>> UPW_;
    MASSystem mas_;

    void constructor(bool verbose);
};

#endif // FIELDCALCULATORTOTAL_H
