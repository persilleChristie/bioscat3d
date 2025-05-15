#include "MASSystem.h"
#include "CrankNicolson.h"
#include "Constants.h"
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include "FieldCalculatorTotal.h"
#include "UtilsExport.h"

Constants constants;

int main() {
    const char* jsonPath = "../../ComparisonTest/Power_int_test/surfaceParamsNormal.json";

    // Create MASSystem to generate test grid (GP surface)
    MASSystem masSystem("Bump", jsonPath);

    std::cout << "l.18" << std::endl;
    // Generate field calculator (inverse crime)
    FieldCalculatorTotal truefield(masSystem);

    // Generate measure-points
    Eigen::MatrixX3d measurepts(10,3);
    measurepts << 10,10,10,
                  5, 7, 4,
                  1, 1, 1,
                  2, 4, 2,
                  -1, -2, -3,
                  0, 2, 1,
                  1.2, 2.3, 3.4,
                  5.5, 5.5, 5.5,
                  4, 6, 8,
                  9, 5, 6;

    int n = measurepts.rows();
    Eigen::MatrixX3cd Etrue = Eigen::MatrixX3cd::Zero(n,3);
    Eigen::MatrixX3cd Htrue = Eigen::MatrixX3cd::Zero(n,3); 

    std::cout << "l.39" << std::endl;
    // Calculate measured field
    truefield.computeFields(Etrue, Htrue, measurepts);

    // Parameters
    double delta = 0.01;
    double gamma = 1.0;
    int iterations = 1000;

    // Instantiate and run Crank-Nicolson sampler
    
    std::cout << "l.50" << std::endl;
    CrankNicolson crank(Etrue, measurepts, delta, gamma, iterations, jsonPath);
    crank.run();

    std::cout << "\nCrank-Nicolson sampling complete.\n";

    std::ifstream file("Estimate.csv");
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return 1;
    }

    std::vector<double> values;
    std::string line;

    // If the CSV has a header, uncomment the next line to skip it
    // std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        // Assuming the column you want is the first one
        if (std::getline(ss, cell, ',')) {
            try {
                values.push_back(std::stod(cell));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid number: " << cell << std::endl;
            }
        }
    }

    file.close();

    // Convert std::vector to Eigen::VectorXd
    int N = (values.size());
    Eigen::VectorXd estimate(N);
    for (int i = 0; i < N; ++i) {
        estimate(i) = values[i];
    }

    // Optional: print the vector
    std::cout << "Difference:\n " << (estimate - masSystem.getPoints().col(2)) << std::endl;
    std::cout << "Mean difference: " << (estimate - masSystem.getPoints().col(2)).sum() / N << std::endl;
    std::cout << "Max difference: " << (estimate - masSystem.getPoints().col(2)).maxCoeff() << std::endl;
    std::cout << "\n\n Relative difference:\n " << abs((estimate - masSystem.getPoints().col(2)).array())/masSystem.getPoints().col(2).array() << std::endl;



    return 0;
}