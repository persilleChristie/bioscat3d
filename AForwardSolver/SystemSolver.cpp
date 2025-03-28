#include "SystemSolver.h"
#include <Eigen/Dense>
#include <iostream>

Eigen::VectorXcd SystemSolver::solve(const Eigen::MatrixXcd& A, const Eigen::VectorXcd& b) {
    if (A.rows() != b.rows()) {
        std::cerr << "Error: Matrix A and vector b have incompatible dimensions." << std::endl;
        return Eigen::VectorXcd();
    }

    return A.fullPivLu().solve(b);
}