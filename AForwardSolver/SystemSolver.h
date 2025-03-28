#ifndef SYSTEMSOLVER_H
#define SYSTEMSOLVER_H

#include <Eigen/Dense>

class SystemSolver {
public:
    static Eigen::VectorXcd solve(const Eigen::MatrixXcd& A, const Eigen::VectorXcd& b);
};

#endif // SYSTEMSOLVER_H
