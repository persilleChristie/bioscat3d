#include "MASSystem.h"
#include "CrankNicolson.h"
#include "Constants.h"
#include <Eigen/Dense>
#include <complex>
#include <iostream>

Constants constants;

int main() {
    const char* jsonPath = "../../ComparisonTest/Power_int_test/surfaceParamsNormal.json";

    // Create MASSystem to generate test grid (GP surface)
    MASSystem masSystem("Bump", jsonPath);
    Eigen::MatrixX3d data_points = masSystem.getPoints();

    // Generate dummy complex data matching surface height
    Eigen::MatrixX3cd data(data_points.rows(), 3);
    for (int i = 0; i < data_points.rows(); ++i) {
        std::complex<double> value = std::complex<double>(data_points(i, 2), 0.0); // Using z as real part
        data.row(i) = Eigen::RowVector3cd(value, value, value);
    }

    // Parameters
    double delta = 0.01;
    double gamma = 1.0;
    int iterations = 10;

    // Instantiate and run Crank-Nicolson sampler
    CrankNicolson crank(data, data_points, delta, gamma, iterations, jsonPath);
    crank.run();

    std::cout << "\n✅ Crank-Nicolson sampling complete.\n";

    return 0;
}