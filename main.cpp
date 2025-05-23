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
    // Flags for tests (max one true)
    bool SurfacePointsTest = false;
    bool PlaneTest = false;
    bool FullSurfaceTest = false;
    
    if (SurfacePointsTest){
        std::cout << std::endl;
        std::cout << "TESTING SURFACE POINTS" << std::endl << std::endl;
    } else if (PlaneTest){
        std::cout << std::endl;
        std::cout << "TESTING TANGENTIAL COMPONENTS FOR PLANE" << std::endl;
        std::cout << "  (only one lambda, one polarization)" << std::endl << std::endl;
    }  else if (FullSurfaceTest){
        std::cout << std::endl;
        std::cout << "TESTING TANGENTIAL COMPONENTS FOR BUMP SURFACE" << std::endl;
        std::cout << "     (only one lambda, one polarization)" << std::endl << std::endl;
    } else {    
        std::cout << std::endl;
        std::cout << "RUNNING TANGENTIAL COMPONENTS FOR FULL EXAMPLE BUMP SURFACE" << std::endl << std::endl;
    }

    // ------------- Load json file --------------
    const char* jsonPath = "../json/surfaceParamsNormalNewGeom_A.json";

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
    double lambda_min = doc["minLambda"].GetDouble();
    double lambda_max = doc["maxLambda"].GetDouble();
    int lambda_nr = doc["lambda_n"].GetInt();

    if (SurfacePointsTest){
        lambda_max = lambda_min;
        lambda_nr = 1;
    }

    if (PlaneTest || FullSurfaceTest){
        lambda_min = lambda_max;
        lambda_nr = 1;
        B = 1;
    }

    // Print lambda_min, lambda_max and lambda_nr
    std::cout << "Lambda min: " << lambda_min << std::endl;
    std::cout << "Lambda max: " << lambda_max << std::endl;
    std::cout << "Lambda nr: " << lambda_nr << std::endl;

    Eigen::VectorXd lambdas = Eigen::VectorXd::LinSpaced(lambda_nr, lambda_min, lambda_max);

    // ------------- Decide highest number of auxilliary points --------------
    int N_fine = static_cast<int>(std::ceil(sqrt(2) * constants.auxpts_pr_lambda * dimension / lambda_min));
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

    if (SurfacePointsTest || FullSurfaceTest){
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
        auto X_np = wrap_eigen(X_fine);
        auto Y_np = wrap_eigen(Y_fine);
        auto Z_np = wrap_eigen(Z_fine);

        // Instantiate Python class
        spline = SplineClass(X_np, Y_np, Z_np);


    } catch (const py::error_already_set &e) {
        std::cerr << "Python error: " << e.what() << "\n";

        return 1;
    }

    int count = 0;

    if (SurfacePointsTest){
        MASSystem mas(spline, lambdas[0], dimension, k, beta_vec);

        Export::saveSurfaceDataCSV("../CSV/surface_data_PN.csv", 
                                            mas.getPoints(), mas.getTau1(), mas.getTau2(), mas.getNormals());
        Export::saveSurfaceDataCSV("../CSV/surface_data_inneraux_PN.csv",
                                            mas.getIntPoints(), mas.getAuxTau1(), mas.getAuxTau2(), mas.getAuxNormals());
        Export::saveSurfaceDataCSV("../CSV/surface_data_outeraux_PN.csv",
                                            mas.getExtPoints(), mas.getAuxTau1(), mas.getAuxTau2(), mas.getAuxNormals());
    }
    else {
        for (auto lambda : lambdas){
            std::cout << "Running MASSystem with lambda: " << lambda << std::endl;

            MASSystem mas(spline, lambda, dimension, k, beta_vec);
            // print variables from MASSystem - for points and tangents only print a few
            std::cout << "Points: " << mas.getPoints().block(0, 0, 5, 3) << std::endl;
            std::cout << "Tangents1: " << mas.getTau1().block(0, 0, 5, 3) << std::endl;
            std::cout << "Tangents2: " << mas.getTau2().block(0, 0, 5, 3) << std::endl;
            std::cout << "Control points: " << mas.getControlPoints().block(0, 0, 5, 3) << std::endl;
            std::cout << "Control tangents1: " << mas.getControlTangents1().block(0, 0, 5, 3) << std::endl;
            std::cout << "Control tangents2: " << mas.getControlTangents2().block(0, 0, 5, 3) << std::endl;
            std::cout << "Auxiliary points (interior): " << mas.getIntPoints().block(0, 0, 5, 3) << std::endl;
            std::cout << "Auxiliary points (exterior): " << mas.getExtPoints().block(0, 0, 5, 3) << std::endl;
            std::cout << "Auxiliary tangents1: " << mas.getAuxTau1().block(0, 0, 5, 3) << std::endl;
            std::cout << "Auxiliary tangents2: " << mas.getAuxTau2().block(0, 0, 5, 3) << std::endl;
            std::cout << "Incidence vector: " << mas.getInc().first.transpose() << std::endl;
            std::cout << "Polarizations: " << mas.getPolarizations().transpose() << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            std::cout << "Running FieldCalculatorTotal..." <<  std::endl;
            FieldCalculatorTotal field(mas);

            auto [error1, error2] = field.computeTangentialError(0);

            Export::saveRealVectorCSV("../CSV/tangential_error1_" + std::to_string(count) + ".csv", error1);
            Export::saveRealVectorCSV("../CSV/tangential_error2_" + std::to_string(count) + ".csv", error2);
            
            
            
            count++;
        }
    }

    return 0;
}

