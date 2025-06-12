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

namespace py = pybind11;
using MatrixRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; // Define RowMajer matrix in Eigen

Constants constants;


/// @brief Translate Eigen matrix into numpy array without letting python own memory
/// @param mat Eigen matrix 
/// @return pybind11 array_t<double>
// py::array_t<double> wrap_eigen(Eigen::MatrixXd& mat) {
//     return py::array_t<double>(
//         {mat.rows(), mat.cols()},
//         {sizeof(double) * mat.cols(), sizeof(double)},
//         mat.data(),
//         py::none() // Do not let Python own the matrix 
//     );
// }

// py::array_t<double> wrap_eigen(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat) {
//     return py::array_t<double>(
//         {mat.rows(), mat.cols()},
//         {sizeof(double) * mat.cols(), sizeof(double)},  // row-major
//         mat.data(),
//         py::none()
//     );
// }


int main() {
    // Choose one true value amongs these
    constexpr bool Surface0 = true;
    constexpr bool Surface1 = false; 
    constexpr bool Surface10 = false;

    // Choose one true value amongs these
    constexpr bool radius01 = false;
    constexpr bool radius1 = true;
    constexpr bool radius10 = false;

    const char* jsonPath = "...";
    std::string fileex = "...";
    if (Surface0){
        std::cout << std::endl;
        std::cout << "TESTING PLANE (ZERO BUMPS)" << std::endl << std::endl;
        jsonPath = "../json/surfaceParamsZero.json";
        fileex = "Zero";
    } else if (Surface1){
        std::cout << std::endl;
        std::cout << "TESTING FOR ONE BUMP SURFACE" << std::endl << std::endl;
        jsonPath = "../json/surfaceParamsOne.json";
        fileex = "One";
    }  else if (Surface10){
        std::cout << std::endl;
        std::cout << "TESTING FOR TEN BUMP SURFACE" << std::endl << std::endl;
        jsonPath = "../json/surfaceParamsTen.json";
        fileex = "Ten";
    } else {    
        std::cout << std::endl;
        std::cout << "RUNNING NO TESTS, CHANGE SURFACE TRUE/FALSE VALUES" << std::endl << std::endl;
        return 1;
    }

    if (radius1){
        std::cout << std::endl;
        std::cout << "TESTING FOR RADIUS 1" << std::endl << std::endl;
        fileex += "_014";
    } else if (radius01) {
        std::cout << std::endl;
        std::cout << "TESTING FOR RADIUS 0.1" << std::endl << std::endl;
        fileex += "_14";
    } else if (radius10) {
        std::cout << std::endl;
        std::cout << "TESTING FOR RADIUS 10" << std::endl << std::endl;
        fileex += "_0014";} else if (radius10) {
        std::cout << std::endl;
        std::cout << "TESTING FOR RADIUS 10" << std::endl << std::endl;
        fileex += "_0014";
    } else {
        std::cout << std::endl;
        std::cout << "RUNNING NO TESTS, CHANGE RADIUS TRUE/FALSE VALUES" << std::endl << std::endl;
        return 1;
    }

    // ------------- Load json file --------------
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
    
    // Load dimension
    const double half_dim = doc["halfWidth_x"].GetDouble();
    const double dimension = 2 * half_dim;
    // Print dimension
    std::cout << "Dimension: " << dimension << std::endl;
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
    const int B = betas.Size();
    std::cout << "Number of polarizations: " << B << std::endl;
    std::cout << "Beta vector: ";
    for (rapidjson::SizeType i = 0; i < betas.Size(); ++i) {
        std::cout << betas[i].GetDouble() << " ";
    }
    std::cout << std::endl;
    // Create eigen vector for betas
    // Note: Eigen::VectorXd is a dynamic size vector, so we can use it directly
    Eigen::VectorXd beta_vec(B);
    for (int i = 0; i < B; ++i) {
        beta_vec(i) = betas[i].GetDouble();   
    }
    // Read lambdas
    const double lambda = doc["maxLambda"].GetDouble();
    const double lambda_fine = lambda;
    constants.setWavelength(lambda);

    // ------------- Decide highest number of auxilliary points --------------
    const int N_fine = static_cast<int>(std::ceil(sqrt(2) * constants.auxpts_pr_lambda * dimension / lambda_fine));
    std::cout << "Number of fine points: " << N_fine << std::endl;

    // ------------- Generate grid -------------
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N_fine, -half_dim, half_dim);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(N_fine, -half_dim, half_dim);

    MatrixRM X_fine(N_fine, N_fine), Y_fine(N_fine, N_fine);
    for (int i = 0; i < N_fine; ++i) {  // Works as meshgrid
        X_fine.row(i) = x.transpose();
        Y_fine.col(i) = y;
    }
    MatrixRM Z_fine = Eigen::MatrixXd::Constant(N_fine, N_fine, 1); // Use plain surface at z=1 as default

    if (Surface1|| Surface10){
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

    // Save grid to CSV
    // Export::saveRealMatrixCSV("../CSV/PN/xfine_" + fileex + ".csv", X_fine);
    // Export::saveRealMatrixCSV("../CSV/PN/yfine_" + fileex + ".csv", Y_fine);
    // Export::saveRealMatrixCSV("../CSV/PN/zfine_" + fileex + ".csv", Z_fine);
    
    py::scoped_interpreter guard{}; // Start Python interpreter
    // ------------ Run python code ---------------
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(1, ".");  // Add local dir to Python path

    // Declare variables needed outside the try-statement
    py::object spline;
    py::array_t<double> X_np;
    py::array_t<double> Y_np;
    py::array_t<double> Z_np;

    try {
        // Import the Python module
        py::module spline_module = py::module::import("Spline");
        // Get the class
        py::object SplineClass = spline_module.attr("Spline");
        // Wrap Eigen matrices as NumPy arrays (shared memory, no copy)
        // auto X_np = wrap_eigen(X_fine);
        // auto Y_np = wrap_eigen(Y_fine);
        // auto Z_np = wrap_eigen(Z_fine);
        X_np = PybindUtils::eigen2numpy(X_fine);
        Y_np = PybindUtils::eigen2numpy(Y_fine);
        Z_np = PybindUtils::eigen2numpy(Z_fine);
        // Instantiate Python class
        spline = SplineClass(X_np, Y_np, Z_np, 0.5);
    } catch (const py::error_already_set &e) {
        std::cerr << "Python error: " << e.what() << "\n";
        return 1;
    }


    // ------------ Create MAS sytem ---------------
    MASSystem mas(spline, dimension, k, beta_vec);

    Export::saveSurfaceDataCSV("../CSV/PN/surface" + fileex + ".csv", 
                                    mas.getPoints(), mas.getTau1(), mas.getTau2(), mas.getNormals());
    Export::saveSurfaceDataCSV("../CSV/PN/surface" + fileex + "_inneraux.csv",
                                    mas.getIntPoints(), mas.getAuxTau1(), mas.getAuxTau2(), mas.getAuxNormals());
    Export::saveSurfaceDataCSV("../CSV/PN/surface" + fileex + "_outeraux.csv",
                                    mas.getExtPoints(), mas.getAuxTau1(), mas.getAuxTau2(), mas.getAuxNormals());
    
    std::cout << std::endl << "Running FieldCalculatorTotal..." << std::endl << std::endl;
    
    FieldCalculatorTotal field(mas, true);

    auto errors = field.computeTangentialError(0);

    Export::saveRealVectorCSV("../CSV/E_tangential_error1_" + fileex + ".csv", errors[0]);
    Export::saveRealVectorCSV("../CSV/E_tangential_error2_" + fileex + ".csv", errors[1]);
    Export::saveRealVectorCSV("../CSV/H_tangential_error1_" + fileex + ".csv", errors[2]);
    Export::saveRealVectorCSV("../CSV/H_tangential_error2_" + fileex + ".csv", errors[3]);

    return 0;
}