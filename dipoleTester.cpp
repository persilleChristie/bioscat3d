#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <memory>
#include "../../lib/Forward/FieldCalculatorDipole.h"
#include "../../lib/Utils/Constants.h"

Constants constants;  // Global constants

int main() {
    // 1. Define dipole
    Eigen::Vector3d dipolePos(0.0, 0.0, 0.0);
    Eigen::Vector3d dipoleDir(1.0, 0.0, 0.0);  // x-directed dipole
    Dipole d(dipolePos, dipoleDir);
    FieldCalculatorDipole dipoleCalc(d, true);  // 'true' → interior dipole

    // 2. Evaluation points along z-axis
    const int N = 5;
    Eigen::MatrixX3d evalPoints(N, 3);
    for (int i = 0; i < N; ++i) {
        evalPoints.row(i) = Eigen::Vector3d(0.0, 0.0, 0.1 * (i + 1));
    }

    // 3. Output fields
    Eigen::MatrixX3cd E, H;
    dipoleCalc.computeFields(E, H, evalPoints, 0);

    // 4. Print results
    std::cout << "Evaluation Points and Dipole Fields:\n";
    for (int i = 0; i < E.rows(); ++i) {
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
