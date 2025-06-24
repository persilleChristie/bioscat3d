#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include "rapidjson/document.h" 
#include "rapidjson/istreamwrapper.h"
#include "Cpp/lib/Utils/Constants.h"
#include "Cpp/lib/Utils/ConstantsModel.h"
#include "Cpp/lib/Forward/MASSystem.h"
#include "Cpp/lib/Forward/FieldCalculatorTotal.h"
#include "Cpp/lib/Utils/UtilsExport.h"
#include "Cpp/lib/Utils/UtilsPybind.h"
#include "Cpp/lib/Inverse/CrankNicolson.h"

Constants constants;
ConstantsModel constantsModel;

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
    int N_fine = static_cast<int>(std::ceil(sqrt(2) * constantsModel.getAuxPtsPrLambda() * dimension / (0.3 * lambda)));
    std::cout << "Number of fine points: " << N_fine << std::endl;


    // ------------- Generate grid -------------
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N_fine, -half_dim, half_dim);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(N_fine, -half_dim, half_dim);

    Eigen::MatrixXd X_fine(N_fine, N_fine), Y_fine(N_fine, N_fine);
    for (int i = 0; i < N_fine; ++i) {  // Works as meshgrid
        X_fine.row(i) = x.transpose();
        Y_fine.col(i) = y;
    }

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

    MASSystem mas(spline, dimension, k, beta_vec);

    // Generate field calculator (inverse crime)
    FieldCalculatorTotal truefield(mas, true);

    // Defining surface
    Eigen::Vector3d corner(-10, -10, 10);
    Eigen::Vector3d basis1(1, 0, 0);
    Eigen::Vector3d basis2(0, 1, 0);
    SurfacePlane surface(corner, basis1, basis2, 10, 10, 50);

    // Calculate measured field
    double power = truefield.computePower(surface)(0);

    // Parameters
    double delta = 0.01;
    double gamma = 100000.0;
    int iterations = 1500;
    std::string kernel_type = "SE";

    CrankNicolson pcn(dimension, k, beta_vec[0], power, surface, kernel_type, delta, gamma, iterations);

    double l = 0.1;
    double tau = 0.1;
    Eigen::MatrixXd estimate = pcn.run(l, tau, true);


    return 0;
}