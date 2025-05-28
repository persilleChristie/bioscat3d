#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include "../../lib/Forward/FieldCalculatorUPW.h"
#include "../../lib/Utils/Constants.h"

Constants constants;  // Define the global constants instance

int main() {
    // Parameters
    Eigen::Vector3d k_vec(0.0, 0.0, -1.0);  // UPW propagating in -z
    double E0 = 1.0;
    double polarization = 0.0;  // beta = 0 → perpendicular

    constants.setWavelength(0.7);

    // Create field calculator
    FieldCalculatorUPW upw(k_vec, E0, polarization);
    

    // Define evaluation points along -z
    const int N = 5;
    Eigen::MatrixX3d evalPoints(N, 3);
    for (int i = 0; i < N; ++i) {
        evalPoints.row(i) = Eigen::Vector3d(0.0, 0.0, -0.1 * i);
    }

    evalPoints.row(0) = Eigen::Vector3d(3,7,11);

    // Compute fields
    // initialize and set dimentions of output matrices
    Eigen::MatrixX3cd E(N, 3), H(N, 3);
    E.setZero();
    H.setZero();
    upw.computeFields(E, H, evalPoints, 0);

    std::cout << "E.rows(): " << E.rows() << ", E.cols(): " << E.cols() << std::endl;
    std::cout << "H.rows(): " << H.rows() << ", H.cols(): " << H.cols() << std::endl;


    if (E.rows() != N || H.rows() != N) {
        std::cerr << "Field dimensions do not match number of points!" << std::endl;
        return 1;
    }

    // Print fields
    std::cout << "Evaluation Points and Fields:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "Point: " << evalPoints.row(i).transpose() << "\n";
        std::cout << "  E: [";
        for (int j = 0; j < 3; ++j)
            std::cout << E(i, j) << (j < 2 ? ", " : "");
        std::cout << "]\n";

        std::cout << "  H: [";
        for (int j = 0; j < 3; ++j)
            std::cout << H(i, j) << (j < 2 ? ", " : "");
        std::cout << "]\n";
        std::cout << "-----------------------------\n";
    }

    return 0;
}
