#include "FieldCalculatorTotal.h"
#include "FieldCalculator.h"
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include "Constants.h"
#include "SystemAssembler.h"
#include "UtilsSolvers.h"
#include "UtilsExport.h"

FieldCalculatorTotal::FieldCalculatorTotal(
    const MASSystem masSystem,
    Constants constants
) : constants_(constants)
{
    constructor(masSystem);
}

void FieldCalculatorTotal::constructor(const MASSystem masSystem)
{
    // ----------- CREATE DIPOLES --------------
    std::vector<std::shared_ptr<FieldCalculator>> sources_int;
    std::vector<std::shared_ptr<FieldCalculator>> sources_ext;

    Eigen::MatrixXd aux_int = masSystem.getInterior();
    Eigen::MatrixXd aux_ext = masSystem.getExterior();
    std::vector<int> aux_idx = masSystem.getIndecees();

    Eigen::MatrixX3d t1 = masSystem.getTau1();
    Eigen::MatrixX3d t2 = masSystem.getTau2();
    Eigen::MatrixX3d points = masSystem.getPoints();

    int M      = points.rows();
    int Nprime = static_cast<int>(aux_int.rows());
    int N      = Nprime * 2;

    int test_index;

    for (int i = 0; i < Nprime; ++i){
        test_index = aux_idx[i];

        sources_int.emplace_back(std::make_shared<FieldCalculatorDipole>(
                                Dipole(points.row(test_index), t1.row(test_index)), constants_, true));
        sources_int.emplace_back(std::make_shared<FieldCalculatorDipole>(
                                Dipole(points.row(test_index), t2.row(test_index)), constants_, true));
 
        sources_ext.emplace_back(std::make_shared<FieldCalculatorDipole>(
                                Dipole(points.row(test_index), t1.row(test_index)), constants_, false));
        sources_ext.emplace_back(std::make_shared<FieldCalculatorDipole>(
                                Dipole(points.row(test_index), t1.row(test_index)), constants_, false));
    }

    // Save dipoles for calculating total field
    this->dipoles_ = sources_int;

    // -------- SOLVE SYSTEM ---------
    // Allocate space for system matrix and UPW
    Eigen::MatrixXcd A(4 * M, 2 * N); 
    Eigen::VectorXcd b(4 * M);
    std::shared_ptr<FieldCalculatorUPW> UPW;

    // Load polarizations and incidence vector
    Eigen::Vector3d kinc = masSystem.getKinc();
    Eigen::VectorXd polarizations = masSystem.getPolarizations();

    int B = polarizations.size();

    // Allocate space for amplitudes
    Eigen::MatrixXcd amplitudes(B, N);

    for (int i = 0; i < B; ++i){
        UPW = std::make_shared<FieldCalculatorUPW>(kinc, 1.0, polarizations(i), constants_);

        SystemAssembler::assembleSystem(A, b, points, t1, t2, sources_int, sources_ext, UPW);

        Eigen::BDCSVD<Eigen::MatrixXcd> svd1(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXcd amps = svd1.solve(b);

        amplitudes.row(i) = amps.head(N);

    }


    this->amplitudes_ = amplitudes;
}


void FieldCalculatorTotal::computeFields(
    Eigen::MatrixX3cd& outE,
    Eigen::MatrixX3cd& outH,
    const Eigen::MatrixX3d& evalPoints,
    int polarization_idx
) const {
    int N = evalPoints.rows();

    Eigen::MatrixX3cd Ei(N, 3), Hi(N, 3);

    for (size_t i = 0; i < dipoles_.size(); ++i) {
        dipoles_[i]->computeFields(Ei, Hi, evalPoints);
        outE += amplitudes_.row(polarization_idx)(i) * Ei;
        outH += amplitudes_.row(polarization_idx)(i) * Hi;
    }
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> FieldCalculatorTotal::computePower(
    const Surface& surface
){
    Eigen::MatrixX3d points = surface.getPoints();
    Eigen::MatrixX3d normals = surface.getNormals();
    int N = points.rows();

    int B = amplitudes_.rows();

    // We assume uniform grid distances
    double dx = (points.row(1) - points.row(0)).norm();
    double dA = dx * dx;
    
    Eigen::MatrixX3cd outE, outH;
    Eigen::VectorXd integral_vec(B);
    double integrand, integral;
    Eigen::Vector3d cross;

    int grid_size = sqrt(N);
    Eigen::MatrixXd integrand_mat(grid_size, grid_size);
    int row, col;

    for (int j = 0; j < B; ++j){
        outE = Eigen::MatrixX3cd::Zero(N,3);
        outH = Eigen::MatrixX3cd::Zero(N,3);

        computeFields(outE, outH, points, j);
        integral = 0.0;  

        for (int i = 0; i < N; ++i){
            cross = 0.5 * outE.row(i).cross(outH.row(i).conjugate()).real();
            integrand = cross.dot(normals.row(i));

            if (j == 0){
                col = i % grid_size;
                row = i / grid_size;

                integrand_mat(row, col) = integrand; 
            }

            integral += integrand * dA;
        }

        integral_vec(j) = integral;

    }

    return {integral_vec, integrand_mat};    
}
