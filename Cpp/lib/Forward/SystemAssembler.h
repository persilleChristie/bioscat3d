#ifndef SYSTEM_ASSEMBLER_H
#define SYSTEM_ASSEMBLER_H

#include "FieldCalculator.h"
#include <Eigen/Dense>
#include <memory>

/// @brief Class for assembling the system of equations for the Method of Auxiliary Sources (MAS).
/// @details This class provides a static method to assemble the system matrix and right-hand side vector
/// for the MAS based on the contributions from various sources, including interior and exterior dipoles,
/// and an incident field. It combines contributions from multiple sources to form a complete system
/// for solving the electromagnetic field equations.
class SystemAssembler {
public:
    /// @brief Assembles the system matrix and right-hand side vector for the MAS.
    /// @details This method constructs the system matrix `A` and the right-hand side vector `b`
    /// based on the contributions from interior and exterior sources, as well as the incident field.
    /// @param A Output matrix representing the system of equations (MxN complex numbers).
    /// @param b Output vector representing the right-hand side of the system (M complex numbers).
    /// @param points Matrix of evaluation points (Mx3 real numbers).
    /// @param tau1 Matrix of tangential vector 1 (Mx3 real numbers).
    /// @param tau2 Matrix of tangential vector 2 (Mx3 real numbers).
    /// @param sources_int Vector of shared pointers to interior FieldCalculator sources.
    /// @param sources_ext Vector of shared pointers to exterior FieldCalculator sources.
    /// @param incident Shared pointer to the incident FieldCalculator source.
    static void assembleSystem(
        Eigen::MatrixXcd& A,
        Eigen::VectorXcd& b,
        Eigen::MatrixX3d& points,
        Eigen::MatrixX3d& tau1,
        Eigen::MatrixX3d& tau2,
        const std::vector<std::shared_ptr<FieldCalculator>>& sources_int,
        const std::vector<std::shared_ptr<FieldCalculator>>& sources_ext,
        const std::shared_ptr<FieldCalculator>& incident
    );
};

#endif // SYSTEM_ASSEMBLER_H
