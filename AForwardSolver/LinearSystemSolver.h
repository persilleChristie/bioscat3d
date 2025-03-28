#ifndef LINEARSYSTEMSOLVER_H
#define LINEARSYSTEMSOLVER_H

#include <Eigen/Dense>

class LinearSystemSolver {
public:
    struct Result {
        Eigen::MatrixXcd A;
        Eigen::VectorXcd b;
        Eigen::VectorXcd x;
    };

    static Result solveSystem(double radius,
                              const Eigen::Vector3d& center,
                              int resolution,
                              double k0,
                              double Gamma_r,
                              const Eigen::Vector3d& k_inc,
                              const Eigen::Vector3d& polarization);
};

#endif // LINEARSYSTEMSOLVER_H
