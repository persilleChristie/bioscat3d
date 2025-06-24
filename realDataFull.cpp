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

double callSplineClass(double width, int grid_size,  const Eigen::MatrixXd& data, py::object& spline){

    // ------------- Generate grid -------------
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(grid_size, 0, width);

    Eigen::MatrixXd X_fine = x.replicate(1, grid_size);           // column vector → N rows, N columns
    Eigen::MatrixXd Y_fine = x.transpose().replicate(grid_size, 1); // row vector → N rows, N columns


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
        Z_np = PybindUtils::eigen2numpy(data);

        // Instantiate Python class
        spline = SplineClass(X_np, Y_np, Z_np);

    } catch (const py::error_already_set &e) {
        std::cerr << "Python error: " << e.what() << "\n";

        return 0.0;
    }
}

int main() {
    
    // --------------------. Controlable variables -----------------------
    double radius = 5.0;
    
    double lambda_min = 0.7;
    double lambda_max = 0.8;
    int lambda_nr = 10;

    double beta_min = 0.0;
    double beta_max = constants.pi/2;
    int beta_nr = 2;

    int surface_nr = 1;

    Eigen::Vector3d k (0, 0, -1);

    // Defining surface
    Eigen::Vector3d corner(-10, -10, 10);
    Eigen::Vector3d basis1(1, 0, 0);
    Eigen::Vector3d basis2(0, 1, 0);
    SurfacePlane surface(corner, basis1, basis2, 10, 10, 50);

    // Creating vectors
    Eigen::VectorXd lambdas = Eigen::VectorXd::LinSpaced(lambda_nr, lambda_min, lambda_max);
    Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(beta_nr, beta_min, beta_max);

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


    Eigen::VectorXd x_large = Eigen::VectorXd::LinSpaced(grid_size, 0, width);

    // ------------ Run python code ---------------
    py::scoped_interpreter guard{}; // Start Python interpreter
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(1, ".");  // Add local dir to Python path

    // Declare variables needed outside the try-statement
    py::object spline;


    callSplineClass(width, grid_size, data, spline);


    // ------------ Calculating all fluxes ---------------
    Eigen::MatrixXd fluxes = Eigen::MatrixXd::Zero(lambda_nr, beta_nr);

    int count = 0;

    for (auto lambda : lambdas){
     
        constants.setWavelength(lambda);

        std::cout << std::endl << "-------- Running MASSystem for lambda: " << count + 1  
                                << "/" << lambda_nr << " --------" << std::endl;

        MASSystem mas(spline, width, k, betas);
            
        std::cout << "Running FieldCalculatorArray..." <<  std::endl;
    
        FieldCalculatorTotal field(mas, true);
  
        fluxes.row(count) = field.computePower(surface);
        std::cout << "Flux computed and saved!" << std::endl;

        
        count++;

    };

    std::string radius_ex = "_radius";

    if (radius > 0){
        radius_ex += "Fixed_" + tag(radius);
    } else if (radius < 0) {
        radius_ex += "MC";
    } else {
        radius_ex += "Guard";
    };

    Export::saveRealMatrixCSV("../CSV/FluxRealData/FullEq_surface" + std::to_string(surface_nr) 
                                + radius_ex + "_lambdamin_" + tag(lambda_min) 
                                + "_lambdamax_" + tag(lambda_max) + "_betanr_" + std::to_string(beta_nr) 
                                + ".csv", fluxes);
    

    return 0;
}