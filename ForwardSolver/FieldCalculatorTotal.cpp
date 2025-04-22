#include "FieldCalculatorTotal.h"
#include "FieldCalculator.h"
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include "Constants.h"
#include "SystemAssembler.h"
#include "UtilsSolvers.h"

FieldCalculatorTotal::FieldCalculatorTotal(
    std::string testpoint_file,
    std::string aux_point_file,
    // const Eigen::VectorXcd & amplitudes,
    // const std::vector<std::shared_ptr<FieldCalculator>>& dipoles,
    const std::shared_ptr<FieldCalculator>& UPW
) : UPW_(UPW)
{
    // if (amplitudes_.size() != static_cast<int>(dipoles_.size())) {
    //     throw std::invalid_argument("Amplitudes and dipoles must have the same size.");
    // }
    constructor(testpoint_file, aux_point_file);
}

void FieldCalculatorTotal::constructor(std::string testpoint_file,
    std::string aux_point_file)
{
    Constants constants;

    ///// LOAD TESTPOINTS /////
    // Read test points into an Eigen matrix
    std::ifstream file(testpoint_file);
    std::string line;

    // Skip header
    std::getline(file, line);
    // Temporary storage for raw vectors
    std::vector<Eigen::Vector3d> positions, normals, tangents1, tangents2;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> values;

        // Parse comma-separated values
        while (std::getline(ss, value, ',')) {
            values.push_back(std::stod(value));
        }

        if (values.size() != 12) {
            std::cerr << "Malformed line: " << line << std::endl;
            continue;
        }

        // Store each vector
        positions.emplace_back(values[0], values[1], values[2]);
        normals.emplace_back(values[3], values[4], values[5]);
        tangents1.emplace_back(values[6], values[7], values[8]);
        tangents2.emplace_back(values[9], values[10], values[11]);
    }

    // Allocate Eigen matrices (rows = num points, cols = 3)
    int M = positions.size();
    Eigen::MatrixXd pts(M, 3);
    Eigen::MatrixXd nrmls(M, 3);
    Eigen::MatrixXd t1(M, 3);
    Eigen::MatrixXd t2(M, 3);

    // Fill matrices
    for (int i = 0; i < M; ++i) {
        pts.row(i) = positions[i];
        nrmls.row(i) = normals[i];
        t1.row(i) = tangents1[i];
        t2.row(i) = tangents2[i];
    }
    

    ///// CREATE DIPOLES /////
    std::vector<std::shared_ptr<FieldCalculator>> sources_int;
    std::vector<std::shared_ptr<FieldCalculator>> sources_ext;

    std::ifstream file2(aux_point_file);
    std::string type;
    double x, y, z;
    int test_index;

    // Skip the header
    std::getline(file2, line);

    while (std::getline(file2, line)) {
        std::stringstream ss(line);

        // Read comma-separated values
        std::string value;

        std::getline(ss, type, ',');        // type
        std::getline(ss, value, ',');       // x
        x = std::stod(value);
        std::getline(ss, value, ',');       // y
        y = std::stod(value);
        std::getline(ss, value, ',');       // z
        z = std::stod(value);
        std::getline(ss, value, ',');       // test_index
        test_index = std::stoi(value);

        Eigen::Vector3d point (x,y,z);

        // Conditional logic based on type
        if (type == "int") {
            // std::cout << "INT type line: x = " << x << ", y = " << y << ", z = " << z << ", test_index = " << test_index << "\n";
            sources_int.emplace_back(std::make_shared<FieldCalculatorDipole>(
                            Dipole(point, t1.row(test_index)), constants));
            sources_int.emplace_back(std::make_shared<FieldCalculatorDipole>(
                            Dipole(point, t2.row(test_index)), constants));
            
        } else if (type == "ext") {
            sources_ext.emplace_back(std::make_shared<FieldCalculatorDipole>(
                            Dipole(point, t1.row(test_index)), constants));
            sources_ext.emplace_back(std::make_shared<FieldCalculatorDipole>(
                            Dipole(point, t2.row(test_index)), constants));
        }
    }

    int Nprime = static_cast<int>(sources_int.size());
    int N      = Nprime + static_cast<int>(sources_ext.size());
    std::cout << "Field calc total, l. 132" << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "M: " << M << std::endl;
    Eigen::MatrixXcd A(4 * M, 2 * N);
    Eigen::VectorXcd b(4 * M);
    std::cout << "Field calc total, l. 129" << std::endl;
    // Save dipoles for calculating total field
    this->dipoles_ = sources_int;
    std::cout << "Field calc total, l. 138" << std::endl;
    ///// SOLVE SYSTEM /////
    SystemAssembler::assembleSystem(A, b, pts, t1, t2, sources_int, sources_ext, UPW_);
    std::cout << "Field calc total, l. 141" << std::endl;
    // solve with UtilsSolver
    auto amps = UtilsSolvers::solveQR(A, b);

    this->amplitudes_ = amps.head(Nprime);
    std::cout << "Field calc total, l. 146" << std::endl;
}


void FieldCalculatorTotal::computeFields(
    Eigen::MatrixX3cd& outE,
    Eigen::MatrixX3cd& outH,
    const Eigen::MatrixX3d& evalPoints
) const {
    int N = evalPoints.rows();

    Eigen::MatrixX3cd Ei(N, 3), Hi(N, 3);

    for (size_t i = 0; i < dipoles_.size(); ++i) {
        dipoles_[i]->computeFields(Ei, Hi, evalPoints);
        outE += amplitudes_[i] * Ei;
        outH += amplitudes_[i] * Hi;
    }

    UPW_->computeFields(Ei, Hi, evalPoints);
    outE += Ei;
    outH += Hi;
}

double FieldCalculatorTotal::computePower(
    const Surface& surface
){
    Eigen::MatrixX3d points = surface.getPoints();
    Eigen::MatrixX3d normals = surface.getNormals();
    int N = points.rows();

    // We assume uniform grid distances
    double dx = (points.row(1) - points.row(0)).norm();
    double dA = dx * dx;
    
    Eigen::MatrixX3cd outE, outH;
    outE = Eigen::MatrixX3cd::Zero(N,3);
    outH = Eigen::MatrixX3cd::Zero(N,3);

    computeFields(outE, outH, points);
    std::complex<double> integrand, integral = 0.0;
    Eigen::Vector3cd cross;

    for (int i = 0; i < N; ++i){
        cross = 0.5 * outE.row(i).cross(outH.row(i).conjugate());
        integrand = cross.dot(normals.row(i));

        integral += integrand * dA;
    }

    return integral.real();
}
