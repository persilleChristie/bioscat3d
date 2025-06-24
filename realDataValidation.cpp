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


auto tag = [](double lambda) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3) << lambda;
    std::string s = ss.str();
    s.erase(std::remove(s.begin(), s.end(), '.'), s.end()); // remove decimal
    if (s.size() > 3) s = s.substr(0, 3); // optional: limit to 3 digits
    return s;
};

double callSplineClass(double width, double rough_size, int grid_size, 
                        const Eigen::MatrixXd& data, py::object& spline){
    // --------------------- Divide grid ---------------------------
    Eigen::VectorXd x_large = Eigen::VectorXd::LinSpaced(grid_size, 0, width); 

    double scale = width / rough_size;
    int grid_size_small = std::floor(grid_size/scale);
    double small_dim = x_large(grid_size_small - 1); 

    // ------------- Generate grid -------------
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(grid_size_small, 0, small_dim); 
    Eigen::MatrixXd X_fine = x.replicate(1, grid_size_small);           // column vector → N rows, N columns
    Eigen::MatrixXd Y_fine = x.transpose().replicate(grid_size_small, 1); // row vector → N rows, N columns
    Eigen::MatrixXd Z_fine = data.block(0, 0, grid_size_small, grid_size_small);


    // GOOD: only declare arrays; matrices already declared and filled above
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

    return small_dim;
}

int main() {
    // --------------------. Controlable variables -----------------------
    double radius = 5.0;

    int surface_nr = 1;

    std::string inputPostfix = "One"; // for input json file, surface parameters.
   
    std::string jsonPathStr = "../json/surfaceParams" + inputPostfix + ".json";
    const char* jsonPath = jsonPathStr.c_str();  // Only needed if a C-style string is required

    // ------------- Load json file --------------
    // Open the file
    std::ifstream ifs(jsonPath);
    if (!ifs.is_open()) {
        return false;
    }
    
    // Wrap the input stream
    rapidjson::IStreamWrapper isw(ifs);
    
    rapidjson::Document doc;

    // Parse the JSON
    doc.ParseStream(isw);
    
    if (doc.HasParseError()) {
        return false;
    }
    
    // -------------------- Read values -------------------
    double half_dim = doc["halfWidth_x"].GetDouble();
    double dimension = 2 * half_dim;

    Eigen::Vector3d k;

    // Read incidence vector
    const auto& kjson = doc["k"];

    for (rapidjson::SizeType i = 0; i < kjson.Size(); ++i) {
        k(i) = kjson[i].GetDouble();
    }
    
    // Read polarizations
    const auto& beta_vec = doc["betas"];
    int beta_nr = beta_vec.Size();
    Eigen::VectorXd betas(beta_nr);
    for (int i = 0; i < beta_nr; ++i) {
        betas(i) = beta_vec[i].GetDouble();   
    }

    // Read wavelengths
    double lambda_min = doc["minLambda"].GetDouble();
    double lambda_max = doc["maxLambda"].GetDouble();
    int lambda_nr = 4;


    // Creating vectors
    Eigen::VectorXd lambdas = Eigen::VectorXd::LinSpaced(lambda_nr, lambda_min, lambda_max);

    // Update constants
    constantsModel.setFixedRadius(radius);

    // --------------------- Load file ---------------------------
    std::string filePath = "../Python/RealData/DataSmooth/gaussian_" + std::to_string(surface_nr) + ".txt"; // Replace with your file path
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Failed to open file.\n";
        return 1;
    }

    std::string line;
    double width = 0.0, height = 0.0;
    std::vector<double> temp_values;

    while (std::getline(file, line)) {
        if (line.rfind("# Width:", 0) == 0) {
            std::istringstream iss(line.substr(8));
            iss >> width;
        } else if (line.rfind("# Height:", 0) == 0) {
            std::istringstream iss(line.substr(9));
            iss >> height;
        } else if (!line.empty() && line[0] != '#') {
            std::istringstream iss(line);
            double val;
            while (iss >> val) {
                temp_values.push_back(val * 1e6);  // Convert from meters to µm
            }
        }
    }

    file.close();

    // Infer grid size
    int total_values = static_cast<int>(temp_values.size());
    int grid_size = static_cast<int>(std::sqrt(total_values));
    if (grid_size * grid_size != total_values) {
        std::cerr << "Data is not a square grid.\n";
        return 1;
    }

    // Fill Eigen matrix (row-major)
    Eigen::MatrixXd data(grid_size, grid_size);
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            data(i, j) = temp_values[i * grid_size + j];
        }
    }


    // ------------ Run python code ---------------
    py::scoped_interpreter guard{}; // Start Python interpreter
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(1, ".");  // Add local dir to Python path

    // Declare variables needed outside the try-statement
    py::object spline;


    double actual_dim = callSplineClass(width, dimension, grid_size, data, spline);


    // ------------ Calculating all fluxes ---------------
    Eigen::MatrixXd fluxes = Eigen::MatrixXd::Zero(lambda_nr, beta_nr);

    int count = 0;

    for (auto lambda : lambdas){
     
        constants.setWavelength(lambda);

        std::cout << std::endl << "-------- Running MASSystem for lambda: " << count + 1  
                                << "/" << lambda_nr << " --------" << std::endl;

        MASSystem mas(spline, actual_dim, k, betas);
            
        std::cout << "Running FieldCalculatorTotal..." <<  std::endl;
        FieldCalculatorTotal field(mas, true);

        for (int b = 0; b < beta_nr; ++b){
            std::cout << "Computing tangential fields for beta: " << betas(b) << std::endl;
            auto errors = field.computeTangentialError(b);
            std::cout << "Tangential errors computed." << std::endl;
            
            // Save tangential errors to CSV files          
            Export::saveRealVectorCSV("../CSV/TangentialErrors_RealSurface" + std::to_string(surface_nr) +"/E_tangential_error1_" 
                                            "_lambda" + tag(lambda) + "_beta" + tag(betas(b)) + ".csv", errors[0]);
            Export::saveRealVectorCSV("../CSV/TangentialErrors_RealSurface" + std::to_string(surface_nr) +"/E_tangential_error2_" 
                                            "_lambda" + tag(lambda) + "_beta" + tag(betas(b)) + ".csv", errors[1]);       
            Export::saveRealVectorCSV("../CSV/TangentialErrors_RealSurface" + std::to_string(surface_nr) +"/H_tangential_error1_" 
                                            "_lambda" + tag(lambda) + "_beta" + tag(betas(b)) + ".csv", errors[2]);
            Export::saveRealVectorCSV("../CSV/TangentialErrors_RealSurface" + std::to_string(surface_nr) +"/H_tangential_error2_" 
                                            "_lambda" + tag(lambda) + "_beta" + tag(betas(b)) + ".csv", errors[3]);       

            std::cout << "Saved tangential errors for beta: " << betas(b) << std::endl;
        }

        
        count++;

    };

    return 0;
}