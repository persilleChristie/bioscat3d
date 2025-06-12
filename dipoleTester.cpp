#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <memory>
#include "../../lib/Forward/FieldCalculatorDipole.h"
#include "../../lib/Utils/Constants.h"

Constants constants;  // Global constants

int main() {
    constants.setWavelength(0.7);  // Set wavelength for the dipole
    // 1. Define dipole
    Eigen::Vector3d dipolePos(-1, 0, 0.14);
    Eigen::Vector3d dipoleDir1(0, 1, 0);  // x-directed dipole
    Eigen::Vector3d dipoleDir2(1, 0, 0); // y-directed dipole
    Dipole d1(dipolePos, dipoleDir1);
    Dipole d2(dipolePos, dipoleDir2);
    FieldCalculatorDipole dipoleCalc1(d1, false);  // 'true' → interior dipole
    FieldCalculatorDipole dipoleCalc2(d2, false);

    // 2. Evaluation points along z-axis
    const int N = 5;
    Eigen::MatrixX3d evalPoints(N, 3);
    for (int i = 0; i < N; ++i) {
        evalPoints.row(i) = Eigen::Vector3d(0.0, 0.0, 0.1 * (i + 1));
    }

    evalPoints.row(0) = Eigen::Vector3d(-1, 0, 0);

    // 3. Output fields
    Eigen::MatrixX3cd E1, H1;
    Eigen::MatrixX3cd E2, H2;
    dipoleCalc1.computeFields(E1, H1, evalPoints, 0);
    dipoleCalc2.computeFields(E2, H2, evalPoints, 0);

    // 4. Print results
    std::cout << "Evaluation Points and Dipole Fields:\n";
    for (int i = 0; i < E1.rows(); ++i) {
        std::cout
            << "Point: " << evalPoints.row(i).transpose() << "\n";
        std::cout << "  E1: [";
        for (int j = 0; j < 3; ++j)
            std::cout << E1(i, j) << (j < 2 ? ", " : "");
        std::cout << "]\n";
        std::cout << "  H1: [";
        for (int j = 0; j < 3; ++j)
            std::cout << H1(i, j) << (j < 2 ? ", " : "");
        std::cout << "]\n";
        std::cout << "  E2: [";
        for (int j = 0; j < 3; ++j)
            std::cout << E2(i, j) << (j < 2 ? ", " : "");
        std::cout << "]\n";
        std::cout << "  H2: [";
        for (int j = 0; j < 3; ++j)
            std::cout << H2(i, j) << (j < 2 ? ", " : "");
        std::cout << "]\n";
        std::cout << "-----------------------------\n";
    }


    // std::cout << "Evaluation Points and Dipole Fields:\n";
    // for (int i = 0; i < E.rows(); ++i) {
    //     std::cout << "Point: " << evalPoints.row(i).transpose() << "\n";
    //     std::cout << "  E: [";
    //     for (int j = 0; j < 3; ++j)
    //         std::cout << E(i, j) << (j < 2 ? ", " : "");
    //     std::cout << "]\n";

    //     std::cout << "  H: [";
    //     for (int j = 0; j < 3; ++j)
    //         std::cout << H(i, j) << (j < 2 ? ", " : "");
    //     std::cout << "]\n";
    //     std::cout << "-----------------------------\n";
    // }

    return 0;
}
