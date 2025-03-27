#include "SystemSolver.h"
#include <Eigen/Dense>
#include <iostream>

Eigen::VectorXd SystemSolver::solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    if (A.rows() != b.rows()) {
        std::cerr << "Error: Matrix A and vector b have incompatible dimensions." << std::endl;
        return Eigen::VectorXd();
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(A);
    return solver.solve(b);
}
