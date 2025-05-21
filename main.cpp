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

namespace py = pybind11;

Constants constants;


/// @brief Translate Eigen matrix into numpy array without letting python own memory
/// @param mat Eigen matrix 
/// @return pybind11 array_t<double>
py::array_t<double> wrap_eigen(Eigen::MatrixXd& mat) {
    return py::array_t<double>(
        {mat.rows(), mat.cols()},
        {sizeof(double) * mat.cols(), sizeof(double)},
        mat.data(),
        py::none() // Do not let Python own the matrix 
    );
}


int main() {
    // ------------- Load json file --------------
    const char* jsonPath = "../json/surfaceParamsNormalNewGeom.json";

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

    const auto& bumpData = doc["bumpData"];

    // Read incidence vector
    const auto& kjson = doc["k"];
    Eigen::Vector3d k;
    for (rapidjson::SizeType i = 0; i < kjson.Size(); ++i) {
        k(i) = kjson[i].GetDouble();
    }
    
    // Read polarizations
    const auto& betas = doc["betas"];
    int B = static_cast<int>(betas.Size());

    Eigen::VectorXd beta_vec(B);
    for (int i = 0; i < B; ++i) {
        beta_vec(i) = betas[i].GetDouble();   // constants.pi / 2 - 
    }

    // Read lambdas
    double lambda_min = doc["minLambda"].GetDouble();
    double lambda_max = doc["maxLambda"].GetDouble();
    int lambda_nr = doc["lambda_n"].GetInt();

    Eigen::VectorXd lambdas = Eigen::VectorXd::LinSpaced(lambda_nr, lambda_min, lambda_max);

    // ------------- Decide highest number of auxilliary points --------------
    int N_fine = static_cast<int>(std::ceil(sqrt(2) * constants.auxpts_pr_lambda * dimension / lambda_min));

    // ------------- Generate grid -------------
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N_fine, -half_dim, half_dim);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(N_fine, -half_dim, half_dim);

    Eigen::MatrixXd X_fine(N_fine, N_fine), Y_fine(N_fine, N_fine);
    for (int i = 0; i < N_fine; ++i) {  // Works as meshgrid
        X_fine.row(i) = x.transpose();
        Y_fine.col(i) = y;
    }

    Eigen::MatrixXd Z_fine = Eigen::MatrixXd::Constant(N_fine, N_fine, 0);

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
        auto X_np = wrap_eigen(X_fine);
        auto Y_np = wrap_eigen(Y_fine);
        auto Z_np = wrap_eigen(Z_fine);

        // Instantiate Python class
        spline = SplineClass(X_np, Y_np, Z_np);


    } catch (const py::error_already_set &e) {
        std::cerr << "Python error: " << e.what() << "\n";

        return 1;
    }

    for (auto lambda : lambdas){

        MASSystem mas(spline, lambda, dimension, k, beta_vec);
        auto points = mas.getPoints();

        std::cout << "Lambda: " << lambda << std::endl;
        std::cout << "Size of points: (" << points.rows() << "," << points.cols() << ")" << std::endl;
        std::cout << "----------------------------" << std::endl;

    }
    

    return 0;
}

