#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include "rapidjson/document.h" 
#include "rapidjson/istreamwrapper.h"
#include "Cpp/lib/Utils/Constants.h"
#include "Cpp/lib/Forward/MASSystem.h"
#include "Cpp/lib/Forward/FieldCalculatorTotal.h"
#include "Cpp/lib/Utils/UtilsExport.h"
#include "Cpp/lib/Utils/UtilsPybind.h"
#include "Cpp/lib/Inverse/CrankNicolson.h"

Constants constants;

int main(){
    
        // Choose one true value amongs these
    bool Surface0 = true;
    bool Surface1 = false; 
    bool Surface10 = false;

    const char* jsonPath = "...";
    std::string fileex = "...";
    
    if (Surface0){
        std::cout << std::endl;
        std::cout << "pCN: TESTING PLANE (ZERO BUMPS)" << std::endl << std::endl;
        
        jsonPath = "../json/surfaceParamsZero.json";

    } else if (Surface1){
        std::cout << std::endl;
        std::cout << "pCN: TESTING FOR ONE BUMP SURFACE" << std::endl << std::endl;

        jsonPath = "../json/surfaceParamsOne.json";
        
    }  else if (Surface10){
        std::cout << std::endl;
        std::cout << "pCN: TESTING FOR TEN BUMP SURFACE" << std::endl << std::endl;

        jsonPath = "../json/surfaceParamsTen.json";
    } else {    
        std::cout << std::endl;
        std::cout << "RUNNING NO TESTS, CHANGE SURFACE TRUE/FALSE VALUES" << std::endl << std::endl;
        return 1;
    }

    // Open the file
    std::ifstream ifs(jsonPath);
    if (!ifs.is_open()) {
        std::cerr << "Could not open JSON file.\n";
        return 1;
    }
    
    // Wrap the input stream
    rapidjson::IStreamWrapper isw(ifs);
    
    // Parse the JSON
    rapidjson::Document doc;
    doc.ParseStream(isw);
    
    if (doc.HasParseError()) {
        std::cerr << "Error parsing JSON.\n";
        return 1;
    }
    
    // Load values
    double half_dim = doc["halfWidth_x"].GetDouble();
    double dimension = 2 * half_dim;

    //print dimension
    std::cout << "Dimension: " << dimension << std::endl;

    const auto& bumpData = doc["bumpData"];

    // Read incidence vector
    const auto& kjson = doc["k"];

    Eigen::Vector3d k;
    for (rapidjson::SizeType i = 0; i < kjson.Size(); ++i) {
        k(i) = kjson[i].GetDouble();
    }
    //print k vector
    std::cout << "k vector: " << k.transpose() << std::endl;
    
    // Read polarizations
    const auto& betas = doc["betas"];
    // B is the number of polarizations
    int B = 1; // betas.Size();
    std::cout << "Number of polarizations: " << B << std::endl;

    for (rapidjson::SizeType i = 0; i < betas.Size(); ++i) {
        std::cout << betas[i].GetDouble() << " ";
    }

    Eigen::VectorXd beta_vec(B);
    for (int i = 0; i < B; ++i) {
        beta_vec(i) = betas[i].GetDouble();   
    }

    // Read lambdas
    double lambda = doc["maxLambda"].GetDouble();
    constants.setWavelength(lambda);

    // ------------- Decide highest number of auxilliary points --------------
    int N_fine = static_cast<int>(std::ceil(sqrt(2) * constants.auxpts_pr_lambda * dimension / (0.3 * lambda)));
    std::cout << "Number of fine points: " << N_fine << std::endl;


    // ------------- Generate grid -------------
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N_fine, -half_dim, half_dim);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(N_fine, -half_dim, half_dim);

    Eigen::MatrixXd X_fine(N_fine, N_fine), Y_fine(N_fine, N_fine);
    for (int i = 0; i < N_fine; ++i) {  // Works as meshgrid
        X_fine.row(i) = x.transpose();
        Y_fine.col(i) = y;
    }

    // suggestion:
    // X_fine = x.transpose().replicate(N_fine, 1); // rows = N_fine, cols = N_fine
    // Y_fine = y.replicate(1, N_fine);             // rows = N_f

    Eigen::MatrixXd Z_fine = Eigen::MatrixXd::Constant(N_fine, N_fine, 0);

    if (Surface1 || Surface10){
            // Read bump data
            const auto& bumpData = doc["bumpData"];

            // Iterate over bumpData array
            for (rapidjson::SizeType i = 0; i < bumpData.Size(); ++i) {
                const auto& bump = bumpData[i];
                double x0 = bump["x0"].GetDouble();
                double y0 = bump["y0"].GetDouble();
                double height = bump["height"].GetDouble();
                double sigma = bump["sigma"].GetDouble();
                
                // Add bumps to data
                Z_fine += (height * exp(-((X_fine.array() - x0)*(X_fine.array() - x0) 
                                    + (Y_fine.array() - y0)*(Y_fine.array() - y0)) / (2 * sigma * sigma))).matrix();
                
        }
    }

    // ------------ Run python code ---------------
    // Should this be scoped to ensure Python interpreter is started and stopped correctly?
    py::scoped_interpreter guard{}; // Start Python interpreter
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(1, ".");  // Add local dir to Python path

    // Declare variables needed outside the try-statement
    py::object spline;

    try {
        // Import the Python module
        py::module spline_module = py::module::import("Spline");

        // Get the class
        py::object SplineClass = spline_module.attr("Spline");

        // Wrap Eigen matrices as NumPy arrays (shared memory, no copy)
        auto X_np = PybindUtils::eigen2numpy(X_fine);
        auto Y_np = PybindUtils::eigen2numpy(Y_fine);
        auto Z_np = PybindUtils::eigen2numpy(Z_fine);

        // Instantiate Python class
        spline = SplineClass(X_np, Y_np, Z_np);


    } catch (const py::error_already_set &e) {
        std::cerr << "Python error: " << e.what() << "\n";

        return 1;
    }

    MASSystem mas(spline, lambda, dimension, k, beta_vec);

    // Generate field calculator (inverse crime)
    FieldCalculatorTotal truefield(mas);

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

    // Calculate measured field
    truefield.computeFields(Etrue, Htrue, measurepts);

    // Parameters
    double delta = 0.01;
    double gamma = 100000.0;
    int iterations = 1500;

    CrankNicolson pcn(dimension, lambda, k, beta_vec[0], Etrue, measurepts, delta, gamma, iterations);



    return 0;
}