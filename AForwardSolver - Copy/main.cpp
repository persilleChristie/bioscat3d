#include "LinearSystemSolver.h"
#include "UtilsExport.h"
#include "Constants.h"
#include <iostream>

Constants constants;

int main() {
    using namespace Eigen;

    // Define parameters
    int num_points = 10;
    double radius = 1.0;
    Vector3d center(0.0, 0.0, 1.0);
    double k0 = 2 * constants.pi;
    double Gamma_r = 1.0;
    Vector3d k_inc(0.0, 0.0, -1.0);
    Vector3d polarization(1.0, 0.0, 0.0);

    // Solve the system
    auto result = LinearSystemSolver::solveSystem(radius, center, num_points, k0, Gamma_r, k_inc, polarization); // take surface as input.

    // Output sizes
    std::cout << "Linear system A has size: " << result.A.rows() << " x " << result.A.cols() << std::endl;
    std::cout << "Right-hand side b has size: " << result.b.size() << std::endl;
    std::cout << "Solution y has size: " << result.y.size() << std::endl;

    // Save to files
    Export::saveMatrixCSV("matrix_A_simple.csv", result.A);
    Export::saveVectorCSV("vector_b_simple.csv", result.b);
    Export::saveVectorCSV("solution_y_simple.csv", result.y);

    return 0;
}
