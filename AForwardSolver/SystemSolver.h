#ifndef SYSTEMSOLVER_H
#define SYSTEMSOLVER_H

#include <Eigen/Dense>
#include <vector>

class SystemSolver {
public:
    static Eigen::VectorXd solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
};

#endif // SYSTEMSOLVER_H
