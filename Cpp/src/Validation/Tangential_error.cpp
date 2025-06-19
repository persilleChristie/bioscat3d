#include <optional>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "rapidjson/document.h" 
#include "rapidjson/istreamwrapper.h"
#include "../../lib/Utils/Constants.h"
#include "../../lib/Utils/ConstantsModel.h"
#include "../../lib/Forward/MASSystem.h"
#include "../../lib/Forward/FieldCalculatorTotal.h"
#include "../../lib/Utils/UtilsExport.h"
#include "../../lib/Utils/UtilsPybind.h"
#include <filesystem>


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
    Eigen::MatrixXd Z_fine = Eigen::MatrixXd::Constant(N_fine, N_fine, 0.0);

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
    lambda = lambda * 1e6; // Convert to micrometers
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3) << lambda;
    std::string s = ss.str();
    s.erase(std::remove(s.begin(), s.end(), '.'), s.end()); // remove decimal
    if (s.size() > 3) s = s.substr(0, 3); // optional: limit to 3 digits
    return s;
};



int main(){
    // ------------- Changes to what is plotted --------------
    // Min max nr of auxiliary points
    int min_auxpoints = 2;
    int max_auxpoints = 5;
    std::string postfix = "FixedUnits10";  // for outpu files, tangential errors.
    std::string inputPostfix = "PseudoReal05"; // for input json file, surface parameters.
    double fixedRadius = 1.0;
    // fixedRadius > 0 means that the radius is fixed
    // fixedRadius < 0 means no guard in place for large curvature
    // fixedRadius = 0 means that the radius is calculated from the maximum curvature with a guard
    constantsModel.setFixedRadius(fixedRadius);
    // set file postfix
    

    // json path controlling surface, polarizations and wavelengths
    std::string jsonPathStr = "../json/surfaceParams" + inputPostfix + ".json";
    const char* jsonPath = jsonPathStr.c_str();  // Only needed if a C-style string is required


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

    double lambda_min = lambdas.minCoeff();

    int B = betas.size();


    // ------------- Start python -------------------
    py::scoped_interpreter guard{}; // Start Python interpreter
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(1, ".");  // Add local dir to Python path


    for (int ap = min_auxpoints; ap < max_auxpoints + 1; ++ap){
        
        constantsModel.setAuxPtsPrLambda(ap);

        auto auxpts_pr_lambda = ap;
    
        std::cout << std::endl;
        std::cout << "----------- RUNNING CODE FOR: " << auxpts_pr_lambda <<  " auxiliary points per wavelength -------------" << std::endl << std::endl;
        
        int N_fine = static_cast<int>(std::ceil(sqrt(2) * auxpts_pr_lambda * dimension / lambda_min));
        
        std::cout << "Number of fine points: " << N_fine << std::endl;
        


        // -------------------- Generate surface --------------------
        auto [X_fine, Y_fine, Z_fine] = generateSurface(N_fine, dimension, bumpData);

        // // save html surface plot to file
        // std::string surfacePlotPath = "../HTML/SurfacePlot" + postfix + "From" + inputPostfix + ".html";
        // Export::saveSurfacePlotHTML(X_fine, Y_fine, Z_fine, surfacePlotPath);
        // std::cout << "Surface plot saved to: " << surfacePlotPath << std::endl;
        // // save surface data to CSV file
        // std::string surfaceDataPath = "../CSV/SurfaceData" + postfix + "From" + inputPostfix + ".csv";
        // Export::saveRealMatrixCSV(surfaceDataPath, X_fine, Y_fine, Z_fine);
        // std::cout << "Surface data saved to: " << surfaceDataPath << std::endl;




        // ----------------- Run python code -----------------
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

            return 1;
        }

        std::string dirE = "../CSV/TangentialErrorsE" + postfix + "From" + inputPostfix;
        std::string dirH = "../CSV/TangentialErrorsH" + postfix + "From" + inputPostfix;

        std::filesystem::create_directories(dirE);
        std::filesystem::create_directories(dirH);

        for (auto lambda : lambdas){
            constants.setWavelength(lambda);
            std::cout << "Running MASSystem with lambda: " << lambda << std::endl;

            MASSystem mas(spline, dimension, k, betas);

            std::cout << "Running FieldCalculatorTotal..." <<  std::endl;
            FieldCalculatorTotal field(mas, true);

            // for (int b = 0; b < B; ++b){
            //     std::cout << "Computing tangential fields for beta: " << betas(b) << std::endl;
            //     auto errors = field.computeTangentialError(b);
            //     std::cout << "Tangential errors computed." << std::endl;
            //     // Save tangential errors to CSV files          
            //     Export::saveRealVectorCSV("../CSV/TangentialErrorsE"+ postfix + "From" + inputPostfix +"/E_tangential_error1_auxpts" + std::to_string(ap) + 
            //                                 "_lambda" + tag(lambda) + "_beta" + tag(betas(b)) + ".csv", errors[0]);
            //     Export::saveRealVectorCSV("../CSV/TangentialErrorsE"+ postfix + "From" + inputPostfix +"/E_tangential_error2_auxpts" + std::to_string(ap) + 
            //                                 "_lambda" + tag(lambda) + "_beta" + tag(betas(b)) + ".csv", errors[1]);
            //     Export::saveRealVectorCSV("../CSV/TangentialErrorsH"+ postfix + "From" + inputPostfix +"/H_tangential_error1_auxpts" + std::to_string(ap) + 
            //                                 "_lambda" + tag(lambda) + "_beta" + tag(betas(b)) + ".csv", errors[2]);
            //     Export::saveRealVectorCSV("../CSV/TangentialErrorsH"+ postfix + "From" + inputPostfix +"/H_tangential_error2_auxpts" + std::to_string(ap) + 
            //                                 "_lambda" + tag(lambda) + "_beta" + tag(betas(b)) + ".csv", errors[3]);
            //     std::cout << "Saved tangential errors for beta: " << betas(b) << std::endl;
            // }
            for (int b = 0; b < B; ++b) {
                std::cout << "Computing tangential fields for beta: " << betas(b) << std::endl;
                auto errors = field.computeTangentialError(b);
                std::cout << "Tangential errors computed." << std::endl;

                Export::saveRealVectorCSV(
                    dirE + "/E_tangential_error1_auxpts" + std::to_string(ap) +
                    "_lambda" + tag(lambda) + "_beta" + tag(betas(b)) + ".csv", errors[0]);

                Export::saveRealVectorCSV(
                    dirE + "/E_tangential_error2_auxpts" + std::to_string(ap) +
                    "_lambda" + tag(lambda) + "_beta" + tag(betas(b)) + ".csv", errors[1]);

                Export::saveRealVectorCSV(
                    dirH + "/H_tangential_error1_auxpts" + std::to_string(ap) +
                    "_lambda" + tag(lambda) + "_beta" + tag(betas(b)) + ".csv", errors[2]);

                Export::saveRealVectorCSV(
                    dirH + "/H_tangential_error2_auxpts" + std::to_string(ap) +
                    "_lambda" + tag(lambda) + "_beta" + tag(betas(b)) + ".csv", errors[3]);

                std::cout << "Saved tangential errors for beta: " << betas(b) << std::endl;
            }
        }
    }
    return 0;
}