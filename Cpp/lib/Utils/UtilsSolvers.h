#ifndef UTILS_SOLVERS_H
#define UTILS_SOLVERS_H

#include <Eigen/Dense>
#include <iostream>
#include <string>

/// @file UtilsSolvers.h
/// @brief Utility functions for solving linear systems using Eigen library.
namespace UtilsSolvers {

    using Eigen::MatrixXcd;
    using Eigen::VectorXcd;

    /// Solve linear system using Eigen's FullPivLU decomposition.
    /// @param A The system matrix.
    /// @param b The right-hand side vector.
    /// @param verbose Whether to print solver info (default: false).
    /// @return The solution vector x.
    inline VectorXcd solveLU(const MatrixXcd& A, const VectorXcd& b, bool verbose = false) {
        if (A.rows() != b.rows()) {
            std::cerr << "[solveLU] Dimension mismatch: A.rows() = " << A.rows()
                      << ", b.rows() = " << b.rows() << std::endl;
            return VectorXcd::Zero(A.cols());
        }

        Eigen::FullPivLU<MatrixXcd> solver(A);
        if (!solver.isInvertible()) {
            std::cerr << "[solveLU] Warning: Matrix may be singular or ill-conditioned." << std::endl;
        }

        VectorXcd x = solver.solve(b);
        if (verbose) {
            std::cout << "[solveLU] Solved system using FullPivLU, relative error: "
                      << (A * x - b).norm() / b.norm() << std::endl;
        }

        return x;
    }

    /// Solve linear system using ColPivHouseholderQR decomposition.
    /// @param A The system matrix.
    /// @param b The right-hand side vector.
    /// @param verbose Whether to print solver info (default: false).
    /// @return The solution vector x.
    inline VectorXcd solveQR(const MatrixXcd& A, const VectorXcd& b, bool verbose = false) {
        if (A.rows() == 0 || A.cols() == 0 || b.size() == 0) {
            std::cerr << "[solveQR] Error: received empty matrix or vector." << std::endl;
            return VectorXcd::Zero(A.cols());
        }
    
        if (A.rows() != b.rows()) {
            std::cerr << "[solveQR] Dimension mismatch: A.rows() = " << A.rows()
                      << ", b.rows() = " << b.rows() << std::endl;
            return VectorXcd::Zero(A.cols());
        }
    
        Eigen::ColPivHouseholderQR<MatrixXcd> solver(A);
        VectorXcd x = solver.solve(b);
    
        if (verbose) {
            std::cout << "[solveQR] Solved system using ColPivHouseholderQR, relative error: "
                      << (A * x - b).norm() / b.norm() << std::endl;
        }
    
        return x;
    }
    

    /// Print a short summary of the solution quality.
    inline void reportResidual(const MatrixXcd& A, const VectorXcd& x, const VectorXcd& b, const std::string& label = "") {
        double relError = (A * x - b).norm() / b.norm();
        std::cout << "[Residual] " << label << " Relative error: " << relError << std::endl;
    }

}  // namespace UtilsSolvers

#endif // UTILS_SOLVERS_H
