#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include "../../ForwardSolver/FieldCalculatorDipole.h"
#include "../../ForwardSolver/Constants.h"
#include "../../ForwardSolver/UtilsDipole.h"

void saveToCSV(const Eigen::MatrixXcd& E, const Eigen::MatrixXcd& H, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Write header
    file << "Ex_Re,Ex_Im,Ey_Re,Ey_Im,Ez_Re,Ez_Im,"
         << "Hx_Re,Hx_Im,Hy_Re,Hy_Im,Hz_Re,Hz_Im\n";

    // Write data
    for (int i = 0; i < E.rows(); ++i) {
        file << E(i, 0).real() << "," << E(i, 0).imag() << ","
             << E(i, 1).real() << "," << E(i, 1).imag() << ","
             << E(i, 2).real() << "," << E(i, 2).imag() << ","
             << H(i, 0).real() << "," << H(i, 0).imag() << ","
             << H(i, 1).real() << "," << H(i, 1).imag() << ","
             << H(i, 2).real() << "," << H(i, 2).imag() << "\n";
    }

    file.close();
}

struct Parameter {
    std::string name;
    double value;
};

int main(int argc, char* argv[]){
    // Subtracting file names (first argument is path to program)
    std::string parameter_file = argv[1];
    std::string testpoint_file = argv[2];
    std::string output_file = argv[3];

    // Open the CSV file
    std::ifstream file(parameter_file); // Replace with your CSV filename
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }
    
    std::vector<Parameter> params;
    std::string line;
    
    // Read the first line (column titles) and ignore it
    std::getline(file, line);
    
    // Read the remaining CSV data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string name, value_str;
        if (std::getline(ss, name, ',') && std::getline(ss, value_str, ',')) {
            try {
                double value = std::stod(value_str);
                params.push_back({name, value});
            } catch (const std::exception &e) {
                std::cerr << "Error parsing value for " << name << "\n";
            }
        }
    }
    file.close();
    
    // Initialize Eigen vectors
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();

    // Initialize wavelength
    Constants constants;
    double wavelength;
    
    // Assign values to Eigen vectors
    for (const auto& param : params) {
        if (param.name == "position_x") position(0) = param.value;
        else if (param.name == "position_y") position(1) = param.value;
        else if (param.name == "position_z") position(2) = param.value;
        else if (param.name == "direction_x") direction(0) = param.value;
        else if (param.name == "direction_y") direction(1) = param.value;
        else if (param.name == "direction_z") direction(2) = param.value;
        else if (param.name == "omega") wavelength = 2*constants.pi/param.value;
    }
    constants.setWavelength(wavelength);
    std::cout << "new wavelength should have been set here" << constants.getWavelength() << std::endl;

    
    // Read test points into an Eigen matrix
    std::ifstream matrix_file(testpoint_file); // Replace with your test points CSV filename
    if (!matrix_file.is_open()) {
        std::cerr << "Error opening test points file!" << std::endl;
        return 1;
    }
    
    std::getline(matrix_file, line); // Ignore the first line (column titles)
    std::vector<Eigen::Vector3d> testpoints_vec;
    
    while (std::getline(matrix_file, line)) {
        std::stringstream ss(line);
        std::string value_str;
        double values[3] = {0.0, 0.0, 0.0};
        int col = 0;
        
        while (std::getline(ss, value_str, ',') && col < 3) {
            try {
                values[col] = std::stod(value_str);
            } catch (const std::exception &e) {
                std::cerr << "Error parsing test point value at column " << col << "\n";
                values[col] = 0.0; // Default to zero if parsing fails
            }
            col++;
        }
        
        if (col == 3) { // Ensure full row of 3 values
            testpoints_vec.push_back(Eigen::Vector3d(values[0], values[1], values[2]));
        }
    }
    matrix_file.close();

    int test_vec_size = static_cast<int>(testpoints_vec.size());

    // Convert vector to Eigen matrix
    Eigen::MatrixXd testpoints(test_vec_size, 3);
    for (int i = 0; i < test_vec_size; ++i) {
        testpoints.row(i) = testpoints_vec[i].transpose();
    };


    ////////////////////////////////
    Eigen::MatrixX3cd E(test_vec_size, 3);
    Eigen::MatrixX3cd H(test_vec_size, 3);

    Dipole HD(position, direction);
    FieldCalculatorDipole EH_HD(HD, constants);
    EH_HD.computeFields(E, H, testpoints);

    Eigen::VectorXd calcImpedance(test_vec_size);

    for (int i = 0; i < test_vec_size; ++i) {
        calcImpedance(i) = E.row(i).norm()/H.row(i).norm();
    };

    std::cout << "Calculated impedance: " << calcImpedance << std::endl;

    // SAVE TO CSV
    saveToCSV(E, H, output_file);

    return 0;
}