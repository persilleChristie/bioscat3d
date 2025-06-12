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

Constants constants;


int main() {
    // Flags for tests (max one true)
    bool SurfacePointsTest = false;
    bool PlaneTest = true; // This should run with a flat surface, no surface bumps
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
    // print constants.auxpts_pr_lambda
    std::cout << "Constants auxpts_pr_lambda: " << constants.auxpts_pr_lambda << std::endl;
    std::cout << "Auxiliary points per wavelength: " << constants.auxpts_pr_lambda << std::endl;
    int N_fine = static_cast<int>(std::ceil(sqrt(2) * constants.auxpts_pr_lambda * dimension / lambda_min));
    std::cout << "Number of fine points: " << N_fine << std::endl;

    std::cout << "half_dim: " << half_dim << std::endl;
    std::cout << "auxpts_pr_lambda: " << constants.auxpts_pr_lambda << std::endl;
    std::cout << "N_fine: " << N_fine << std::endl;



    // ------------- Generate grid -------------
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N_fine, -half_dim, half_dim);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(N_fine, -half_dim, half_dim);

    // Eigen::MatrixXd X_fine(N_fine, N_fine), Y_fine(N_fine, N_fine);
    // for (int i = 0; i < N_fine; ++i) {  // Works as meshgrid
    //     X_fine.row(i) = x.transpose();
    //     Y_fine.col(i) = y;
    // }

    Eigen::MatrixXd X_fine = x.replicate(1, N_fine);           // column vector → N rows, N columns
    Eigen::MatrixXd Y_fine = y.transpose().replicate(N_fine, 1); // row vector → N rows, N columns
    Eigen::MatrixXd Z_fine = Eigen::MatrixXd::Constant(N_fine, N_fine, 0.0);

    std::cout << "main line 148" << std::endl;
    std::cout << "X_fine(0, 0): " << X_fine(0, 0) << std::endl;
    std::cout << "Y_fine(0, 0): " << Y_fine(0, 0) << std::endl;
    std::cout << "Z_fine(0, 0): " << Z_fine(0, 0) << std::endl;


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

    std::cout << "Z_fine range: " << Z_fine.minCoeff() << " to " << Z_fine.maxCoeff() << std::endl;
    std::cout << "Z_fine sample: \n" << Z_fine.block(0, 0, 5, 5) << std::endl;

    std::cout << "X_fine (0:5, 0:5) before Python: \n" << X_fine.block(0, 0, 5, 5) << std::endl;
    std::cout << "Y_fine (0:5, 0:5) before Python: \n" << Y_fine.block(0, 0, 5, 5) << std::endl;
    std::cout << "Z_fine (0:5, 0:5) before Python: \n" << Z_fine.block(0, 0, 5, 5) << std::endl;

    // ------------ Run python code ---------------
    // Should this be scoped to ensure Python interpreter is started and stopped correctly?
    py::scoped_interpreter guard{}; // Start Python interpreter
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(1, ".");  // Add local dir to Python path

    // Declare variables needed outside the try-statement
    py::object spline;

    // GOOD: only declare arrays; matrices already declared and filled above
    py::array_t<double> X_np, Y_np, Z_np;

    try {
        // Import the Python module
        py::module spline_module = py::module::import("Spline");
        // Get the class
        py::object SplineClass = spline_module.attr("Spline");

        // Wrap Eigen matrices as NumPy arrays (shared memory, no copy)
        using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        X_np = PybindUtils::eigen2numpy(X_fine);
        Y_np = PybindUtils::eigen2numpy(Y_fine);
        Z_np = PybindUtils::eigen2numpy(Z_fine);


        // Instantiate Python class
        spline = SplineClass(X_np, Y_np, Z_np);


    } catch (const py::error_already_set &e) {
        std::cerr << "Python error: " << e.what() << "\n";

        return 1;
    }

    int count = 0;

    if (SurfacePointsTest){
        constants.setWavelength(lambda_min);
        MASSystem mas(spline, dimension, k, beta_vec);

        Export::saveSurfaceDataCSV("../CSV/PN/surface_data_PN.csv", 
                                            mas.getPoints(), mas.getTau1(), mas.getTau2(), mas.getNormals());
        Export::saveSurfaceDataCSV("../CSV/PN/surface_data_inneraux_PN.csv",
                                            mas.getIntPoints(), mas.getAuxTau1(), mas.getAuxTau2(), mas.getAuxNormals());
        Export::saveSurfaceDataCSV("../CSV/PN/surface_data_outeraux_PN.csv",
                                            mas.getExtPoints(), mas.getAuxTau1(), mas.getAuxTau2(), mas.getAuxNormals());
    
    } 
    else {
        for (auto lambda : lambdas){
            constants.setWavelength(lambda);
            std::cout << "Running MASSystem with lambda: " << lambda << std::endl;

            MASSystem mas(spline, dimension, k, beta_vec);
            // print variables from MASSystem - for points and tangents only print a few
            // std::cout << "Points: " << mas.getPoints().block(0, 0, 5, 3) << std::endl;
            // std::cout << "Tangents1: " << mas.getTau1().block(0, 0, 5, 3) << std::endl;
            // std::cout << "Tangents2: " << mas.getTau2().block(0, 0, 5, 3) << std::endl;
            // std::cout << "Control points: " << mas.getControlPoints().block(0, 0, 5, 3) << std::endl;
            // std::cout << "Control tangents1: " << mas.getControlTangents1().block(0, 0, 5, 3) << std::endl;
            // std::cout << "Control tangents2: " << mas.getControlTangents2().block(0, 0, 5, 3) << std::endl;
            // std::cout << "Auxiliary points (interior): " << mas.getIntPoints().block(0, 0, 5, 3) << std::endl;
            // std::cout << "Auxiliary points (exterior): " << mas.getExtPoints().block(0, 0, 5, 3) << std::endl;
            // std::cout << "Auxiliary tangents1: " << mas.getAuxTau1().block(0, 0, 5, 3) << std::endl;
            // std::cout << "Auxiliary tangents2: " << mas.getAuxTau2().block(0, 0, 5, 3) << std::endl;
            // std::cout << "Incidence vector: " << mas.getInc().transpose() << std::endl;
            // std::cout << "Polarizations: " << mas.getPolarizations().transpose() << std::endl;
            // std::cout << "----------------------------------------" << std::endl;

            std::cout << "Running FieldCalculatorTotal..." <<  std::endl;
            FieldCalculatorTotal field(mas, true);

            auto errors = field.computeTangentialError(0);

            Export::saveRealVectorCSV("../CSV/E_tangential_error1_" + std::to_string(count) + ".csv", errors[0]);
            Export::saveRealVectorCSV("../CSV/E_tangential_error2_" + std::to_string(count) + ".csv", errors[1]);
            Export::saveRealVectorCSV("../CSV/H_tangential_error1_" + std::to_string(count) + ".csv", errors[2]);
            Export::saveRealVectorCSV("../CSV/H_tangential_error2_" + std::to_string(count) + ".csv", errors[3]);
            
            
            
            count++;
        }
    }

    return 0;
}

