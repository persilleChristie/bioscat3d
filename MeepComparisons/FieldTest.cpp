#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include "../ForwardSolver/FieldCalculatorTotal.h"
#include "../ForwardSolver/FieldCalculatorUPW.h"
#include "../ForwardSolver/Constants.h"
#include "../ForwardSolver/UtilsExport.h"

struct Parameter {
    std::string name;
    double value;
};

int main(int argc, char* argv[]){
    if (argc < 7){
        std::cerr << "Usage: " << argv[0] << " <parameter_file> <testpoint_file> <auxpoint_file> <evalpoint_file> <E_output_file> <H_output_file>\n";
        return 1;
    }

    Constants constants;

    std::string parameter_file = argv[1];
    std::string testpoint_file = argv[2];
    std::string auxpoint_file  = argv[3];
    std::string evalpoint_file = argv[4];
    std::string E_output_file  = argv[5];
    std::string H_output_file  = argv[6];

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
    Eigen::Vector3d k_inc = Eigen::Vector3d::Zero();

    // Initialize values
    double wavelength, polarization, E0;
    
    // Assign values
    for (const auto& param : params) {
        if (param.name == "incident_x") k_inc(0) = param.value;
        else if (param.name == "incident_y") k_inc(1) = param.value;
        else if (param.name == "incident_z") k_inc(2) = param.value;
        else if (param.name == "wavelength") wavelength = param.value;
        else if (param.name == "polarization") polarization = param.value;
        else if (param.name == "E0") E0 = param.value;
    }

    constants.setWavelength(wavelength);

    // Load evaluation points into an Eigen matrix
    std::ifstream matrix_file(evalpoint_file); // Replace with your test points CSV filename
    if (!matrix_file.is_open()) {
        std::cerr << "Error opening test points file!" << std::endl;
        return 1;
    }
    
    std::getline(matrix_file, line); // Ignore the first line (column titles)
    std::vector<Eigen::Vector3d> evalpoints_vec;
    
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
            evalpoints_vec.push_back(Eigen::Vector3d(values[0], values[1], values[2]));
        }
    }
    matrix_file.close();

    int evalvec_size = static_cast<int>(evalpoints_vec.size());

    // Convert vector to Eigen matrix
    Eigen::MatrixX3d evalpoints(evalvec_size, 3);
    for (int i = 0; i < evalvec_size; ++i) {
        evalpoints.row(i) = evalpoints_vec[i].transpose();
    };


    // Pre-compute incident field
    std::shared_ptr<FieldCalculatorUPW> incident = std::make_shared<FieldCalculatorUPW>(k_inc, E0, polarization, constants);
    
    // Compute total field
    FieldCalculatorTotal field(testpoint_file, auxpoint_file, incident, constants);

    // Allocate space for field evaluations
    Eigen::MatrixX3cd outE(evalvec_size, 3), outH(evalvec_size, 3);

    // Compute field
    field.computeFields(outE, outH, evalpoints);

    // Save to csv files
    Export::saveMatrixCSV(E_output_file, outE);
    Export::saveMatrixCSV(H_output_file, outH);

    return 0;
}