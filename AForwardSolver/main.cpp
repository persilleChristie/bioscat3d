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
    auto result = LinearSystemSolver::solveSystem(radius, center, num_points, k0, Gamma_r, k_inc, polarization);

    // Output sizes
    std::cout << "Linear system A has size: " << result.A.rows() << " x " << result.A.cols() << std::endl;
    std::cout << "Right-hand side b has size: " << result.b.size() << std::endl;
    std::cout << "Solution x has size: " << result.x.size() << std::endl;

    // Save to files
    saveMatrixCSV("matrix_A_simple.csv", result.A);
    saveVectorCSV("vector_b_simple.csv", result.b);
    saveVectorCSV("solution_y_simple.csv", result.x);

    return 0;
}
