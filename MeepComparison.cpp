#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <string>
#include <vector>
#include <cmath>
#include "rapidjson/document.h" 
#include "rapidjson/istreamwrapper.h"
#include "Cpp/lib/Utils/Constants.h"
#include "Cpp/lib/Utils/ConstantsModel.h"
#include "Cpp/lib/Forward/MASSystem.h"
#include "Cpp/lib/Forward/SurfacePlane.h"
#include "Cpp/lib/Forward/FieldCalculatorTotal.h"
#include "Cpp/lib/Forward/FieldCalculatorArray.h"
#include "Cpp/lib/Utils/UtilsExport.h"
#include "Cpp/lib/Utils/UtilsPybind.h"

Constants constants;
ConstantsModel constantsModel;

bool loadJson(const char* jsonPath, double& dimension,
              Eigen::Vector3d& k,
              Eigen::VectorXd& lambdas,
              Eigen::VectorXd& beta_vec,
              rapidjson::Document& doc){
    // ------------- Load json file --------------
    // Open the file
    std::ifstream ifs(jsonPath);
    if (!ifs.is_open()) {
        return false;
    }
    
    // Wrap the input stream
    rapidjson::IStreamWrapper isw(ifs);
    
    // Parse the JSON
    doc.ParseStream(isw);
    
    if (doc.HasParseError()) {
        return false;
    }
    
    // -------------------- Read values -------------------
    double half_dim = doc["halfWidth_x"].GetDouble();
    dimension = 2 * half_dim;

    std::cout << "Dimension: " << dimension << std::endl;

    // Read incidence vector
    const auto& kjson = doc["k"];

    for (rapidjson::SizeType i = 0; i < kjson.Size(); ++i) {
        k(i) = kjson[i].GetDouble();
    }

    std::cout << "k vector: " << k.transpose() << std::endl;
    
    // Read polarizations
    const auto& betas = doc["betas"];
    int B = betas.Size();
    Eigen::VectorXd beta_vec_loop(B);
    for (int i = 0; i < B; ++i) {
        beta_vec_loop(i) = betas[i].GetDouble();   
    }
    beta_vec = beta_vec_loop;

    // Read wavelengths
    double lambda_min = doc["minLambda"].GetDouble();
    double lambda_max = doc["maxLambda"].GetDouble();
    int lambda_nr = doc["lambda_n"].GetInt();

    std::cout << "Lambda min: " << lambda_min << std::endl;
    std::cout << "Lambda max: " << lambda_max << std::endl;
    std::cout << "Lambda nr: " << lambda_nr << std::endl;

    lambdas = Eigen::VectorXd::LinSpaced(lambda_nr, lambda_min, lambda_max);


    return true;
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> generateSurface(int N_fine, double dimension, rapidjson::Value& bumpData){
    double half_dim = dimension/2.0;
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N_fine, -half_dim, half_dim);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(N_fine, -half_dim, half_dim);

    Eigen::MatrixXd X_fine = x.replicate(1, N_fine);           // column vector → N rows, N columns
    Eigen::MatrixXd Y_fine = y.transpose().replicate(N_fine, 1); // row vector → N rows, N columns
    Eigen::MatrixXd Z_fine = Eigen::MatrixXd::Constant(N_fine, N_fine, -1.5);

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

    return {X_fine, Y_fine, Z_fine};
}

auto tag = [](double lambda) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3) << lambda;
    std::string s = ss.str();
    s.erase(std::remove(s.begin(), s.end(), '.'), s.end()); // remove decimal
    if (s.size() > 3) s = s.substr(0, 3); // optional: limit to 3 digits
    return s;
};


int main() {
    // --------------------. Controlable variables -----------------------
    std::string inputPostfix = "NormalNewGeom"; // for input json file, surface parameters.
    double fixedRadius = 5.0;
    constantsModel.setFixedRadius(fixedRadius);
    

    // json path controlling surface, polarizations and wavelengths
    std::string jsonPathStr = "../json/surfaceParams" + inputPostfix + ".json";
    const char* jsonPath = jsonPathStr.c_str();  // Only needed if a C-style string is required


    // Create surfaces
    double ms = 0.5; // Monitor size 
    Eigen::Vector3d corner(-ms, -ms, -ms);
    Eigen::Vector3d basis1(1, 0, 0);
    Eigen::Vector3d basis2(0, 1, 0);
    SurfacePlane Z1(corner, basis1, basis2, 1, 1, 20);

    corner << -ms, -ms, ms;
    SurfacePlane Z2(corner, basis1, basis2, 1, 1, 20);

    corner << -ms, -ms, -ms;
    basis1 << 1, 0, 0;
    basis2 << 0, 0, 1;
    SurfacePlane Y1(corner, basis1, basis2, 1, 1, 20);

    corner << -ms, ms, -ms;
    SurfacePlane Y2(corner, basis1, basis2, 1, 1, 20);

    corner << -ms, -ms, -ms;
    basis1 << 0, 1, 0;
    basis2 << 0, 0, 1;
    SurfacePlane X1(corner, basis1, basis2, 1, 1, 20);

    corner << ms, -ms, -ms;
    SurfacePlane X2(corner, basis1, basis2, 1, 1, 20);

    // ------------- Load json file --------------
    double dimension;
    Eigen::Vector3d k;
    Eigen::VectorXd lambdas, betas;
    rapidjson::Document doc;

    auto resultOpt = loadJson(jsonPath, dimension, k, lambdas, betas, doc);
    if (!resultOpt) {
        std::cerr << "Error when opening or parsing JSON file.\n";
        return 1;
    }

    auto& bumpData = doc["bumpData"];

    double lambda_max = lambdas.maxCoeff();
    double lambda_min = lambdas.minCoeff();

    int B = betas.size();

    int N_fine = static_cast<int>(std::ceil(sqrt(2) * constantsModel.getAuxPtsPrLambda() * dimension / lambda_min));
        
    // -------------------- Generate surface --------------------
    auto [X_fine, Y_fine, Z_fine] = generateSurface(N_fine, dimension, bumpData);

    // ------------ Run python code ---------------
    py::scoped_interpreter guard{}; // Start Python interpreter
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(1, ".");  // Add local dir to Python path

    // Declare variables needed outside the try-statement
    py::object spline;

    py::array_t<double> X_np, Y_np, Z_np;

    try {
         // Import the Python module
        py::module spline_module = py::module::import("Spline");
        // Get the class
        py::object SplineClass = spline_module.attr("Spline");

        // Wrap Eigen matrices as NumPy arrays (shared memory, no copy)
        X_np = PybindUtils::eigen2numpy(X_fine);
        Y_np = PybindUtils::eigen2numpy(Y_fine);
        Z_np = PybindUtils::eigen2numpy(Z_fine);

        // Instantiate Python class
        spline = SplineClass(X_np, Y_np, Z_np);

    } catch (const py::error_already_set &e) {
        std::cerr << "Python error: " << e.what() << "\n";

        return 0.0;
    }

    double dim = dimension;


    // ------------ Calculating all fluxes ---------------
    int lambda_nr = static_cast<int>(lambdas.size());;
    int beta_nr = static_cast<int>(betas.size());

    Eigen::MatrixXd fluxesX1 = Eigen::MatrixXd::Zero(lambda_nr, beta_nr);
    Eigen::MatrixXd fluxesX2 = Eigen::MatrixXd::Zero(lambda_nr, beta_nr);
    Eigen::MatrixXd fluxesY1 = Eigen::MatrixXd::Zero(lambda_nr, beta_nr);
    Eigen::MatrixXd fluxesY2 = Eigen::MatrixXd::Zero(lambda_nr, beta_nr);
    Eigen::MatrixXd fluxesZ1 = Eigen::MatrixXd::Zero(lambda_nr, beta_nr);
    Eigen::MatrixXd fluxesZ2 = Eigen::MatrixXd::Zero(lambda_nr, beta_nr);


    int count = 0;

    for (auto lambda : lambdas){
     
        constants.setWavelength(lambda);

        std::cout << std::endl << "-------- Running MASSystem for lambda: " << count + 1  
                                << "/" << lambda_nr << " --------" << std::endl;

        MASSystem mas(spline, dim, k, betas);
            
        std::cout << "Running FieldCalculatorTotal..." <<  std::endl;
        FieldCalculatorTotal field(mas, true);

        std::cout << "Calculating fluxes..." << std::endl;
        fluxesZ1.row(count)= field.computePower(Z1);
        fluxesZ2.row(count)= field.computePower(Z2);
        fluxesY1.row(count)= field.computePower(Y1);
        fluxesY2.row(count)= field.computePower(Y2);
        fluxesX1.row(count)= field.computePower(X1);
        fluxesX2.row(count)= field.computePower(X2);
        
        
        count++;

    };

    Export::saveRealMatrixCSV("../CSV/MeepFlux/Flux_Z1_lambdamin_" + tag(lambda_min) 
                                + "_lambdamax_" + tag(lambda_max) + ".csv", fluxesZ1);
    Export::saveRealMatrixCSV("../CSV/MeepFlux/Flux_Z2_lambdamin_" + tag(lambda_min) 
                                + "_lambdamax_" + tag(lambda_max) + ".csv", fluxesZ2);
    Export::saveRealMatrixCSV("../CSV/MeepFlux/Flux_Y1_lambdamin_" + tag(lambda_min) 
                                + "_lambdamax_" + tag(lambda_max) + ".csv", fluxesY1);
    Export::saveRealMatrixCSV("../CSV/MeepFlux/Flux_Y2_lambdamin_" + tag(lambda_min) 
                                + "_lambdamax_" + tag(lambda_max) + ".csv", fluxesY2);
    Export::saveRealMatrixCSV("../CSV/MeepFlux/Flux_X1_lambdamin_" + tag(lambda_min) 
                                + "_lambdamax_" + tag(lambda_max) + ".csv", fluxesX1);
    Export::saveRealMatrixCSV("../CSV/MeepFlux/Flux_X2_lambdamin_" + tag(lambda_min) 
                                + "_lambdamax_" + tag(lambda_max) + ".csv", fluxesX2);

    return 0;
}