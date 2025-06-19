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
                        const Eigen::VectorXd& x_large, const Eigen::MatrixXd& data, py::object& spline){
    // --------------------- Divide grid ---------------------------
    int scale = std::floor(width / rough_size);
    int grid_size_small = std::floor(grid_size/scale);
    double small_dim = x_large(grid_size_small - 1); 
    double half_dim_small = 0.5 * small_dim;


    // ------------- Generate grid -------------
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(grid_size_small, 0, half_dim_small);

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
    double rough_size_min = 0.5;
    double rough_size_max = 1.0;  // Roughly how big the patch should be

    bool variable_patch_size = true;
    
    double lambda_min = 0.25;
    double lambda_max = 0.75;
    int lambda_nr = 20;

    double beta_min = 0.0;
    double beta_max = constants.pi/2;
    int beta_nr = 2;

    int surface_nr = 3;

    Eigen::Vector3d k (0, 0, -1);

    // Defining surface
    Eigen::Vector3d corner(-10, -10, 10);
    Eigen::Vector3d basis1(1, 0, 0);
    Eigen::Vector3d basis2(0, 1, 0);
    SurfacePlane surface(corner, basis1, basis2, 10, 10, 50);

    // Creating vectors
    Eigen::VectorXd lambdas = Eigen::VectorXd::LinSpaced(lambda_nr, lambda_min, lambda_max);
    Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(beta_nr, beta_min, beta_max);
    Eigen::VectorXd rough_sizes = Eigen::VectorXd::LinSpaced(lambda_nr, rough_size_min, rough_size_max);

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


    Eigen::VectorXd x_large = Eigen::VectorXd::LinSpaced(grid_size, 0, width);

    // ------------ Run python code ---------------
    py::scoped_interpreter guard{}; // Start Python interpreter
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(1, ".");  // Add local dir to Python path

    // Declare variables needed outside the try-statement
    py::object spline;
    double small_dim;

    
    if (not variable_patch_size){
        small_dim = callSplineClass(width, rough_size_max, grid_size, x_large, data, spline);
    };


    // ------------ Calculating all fluxes ---------------
    Eigen::MatrixXd fluxes = Eigen::MatrixXd::Zero(lambda_nr, beta_nr);

    int count = 0;

    for (auto lambda : lambdas){
        double rough_size = rough_sizes(count);

        if (variable_patch_size){
            small_dim = callSplineClass(width, rough_size, grid_size, x_large, data, spline);
        };

        if (small_dim <= 0){
            std::cerr << "Dimension error" << "\n";

            return 1;
        };

    
        constants.setWavelength(lambda);

        std::cout << std::endl << "-------- Running MASSystem for lambda: " << count + 1  
                                << "/" << lambda_nr << " --------" << std::endl;

        MASSystem mas(spline, small_dim, k, betas);
            
        std::cout << "Running FieldCalculatorArray..." <<  std::endl;
        FieldCalculatorArray field(mas, width, false);

        fluxes.row(count) = field.computePower(surface);
        std::cout << "Flux computed and saved!" << std::endl;

        
        count++;

    };
    
    std::string nameex = (variable_patch_size) ? "variablePatchSize" : "constantPatchSize_" + tag(rough_size_max);

    Export::saveRealMatrixCSV("../CSV/FluxRealData/ArrayEq_surface" + std::to_string(surface_nr) 
                                + nameex + "_lambdamin_" + tag(lambda_min) 
                                + "_lambdamax_" + tag(lambda_max) + "_betanr_" + std::to_string(beta_nr) 
                                + ".csv", fluxes);
    

    return 0;
}