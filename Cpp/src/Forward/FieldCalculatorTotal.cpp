#include <Eigen/Dense>
#include "../../lib/Forward/SystemAssembler.h"
#include "../../lib/Forward/FieldCalculator.h"
#include "../../lib/Forward/FieldCalculatorTotal.h"
#include "../../lib/Forward/FieldCalculatorDipole.h"
#include "../../lib/Forward/FieldCalculatorUPW.h"
#include "../../lib/Utils/Constants.h"
#include "../../lib/Utils/UtilsDipole.h"

FieldCalculatorTotal::FieldCalculatorTotal(
    const MASSystem masSystem
) : mas_(masSystem)
{
    constructor();
}

void FieldCalculatorTotal::constructor()
{
    // ----------- CREATE DIPOLES --------------
    std::vector<std::shared_ptr<FieldCalculator>> sources_int;
    std::vector<std::shared_ptr<FieldCalculator>> sources_ext;

    Eigen::MatrixXd aux_int = mas_.getIntPoints();
    Eigen::MatrixXd aux_ext = mas_.getExtPoints();
    Eigen::MatrixX3d aux_t1 = mas_.getAuxTau1();
    Eigen::MatrixX3d aux_t2 = mas_.getAuxTau2();

    Eigen::MatrixX3d t1 = mas_.getTau1();
    Eigen::MatrixX3d t2 = mas_.getTau2();
    Eigen::MatrixX3d points = mas_.getPoints();

    int M      = points.rows();
    int Nprime = aux_int.rows();
    int N      = Nprime * 2;

    for (int i = 0; i < Nprime; ++i){
        sources_int.emplace_back(std::make_shared<FieldCalculatorDipole>(
                                Dipole(aux_int.row(i), aux_t1.row(i)), true));
        sources_int.emplace_back(std::make_shared<FieldCalculatorDipole>(
                                Dipole(aux_int.row(i), aux_t2.row(i)), true));
 
        sources_ext.emplace_back(std::make_shared<FieldCalculatorDipole>(
                                Dipole(aux_ext.row(i), aux_t1.row(i)), false));
        sources_ext.emplace_back(std::make_shared<FieldCalculatorDipole>(
                                Dipole(aux_ext.row(i), aux_t2.row(i)), false));
    }

    // Save dipoles for calculating total field
    this->dipoles_ = sources_int;
    this->dipoles_ext_ = sources_ext;

    // -------- SOLVE SYSTEM ---------
    // Allocate space for system matrix and UPW
    Eigen::MatrixXcd A(4 * M, 2 * N); 
    Eigen::VectorXcd b(4 * M);
    std::shared_ptr<FieldCalculatorUPW> UPW;
    std::vector<std::shared_ptr<FieldCalculator>> UPW_list;

    // Load polarizations and incidence vector
    auto [kinc, lambda] = mas_.getInc();
    Eigen::VectorXd polarizations = mas_.getPolarizations();

    int B = polarizations.size();

    // Allocate space for amplitudes
    Eigen::MatrixXcd amplitudes(B, N);
    Eigen::MatrixXcd amplitudes_ext(B, N);

    std::string filename;

    

    for (int i = 0; i < B; ++i){
        UPW = std::make_shared<FieldCalculatorUPW>(kinc, 1.0, polarizations(i));
        UPW_list.emplace_back(UPW);

        SystemAssembler::assembleSystem(A, b, points, t1, t2, sources_int, sources_ext, UPW);

        Eigen::BDCSVD<Eigen::MatrixXcd> svd1(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXcd amps = svd1.solve(b);

        amplitudes.row(i) = amps.head(N);
        amplitudes_ext.row(i) = amps.tail(N);

        // if (i == 0){
        //     Export::saveMatrixCSV("FilesCSV/matrix_A_simple.csv", A);
            
        // }

        // filename = "FilesCSV/solution_y_" + std::to_string(i) + ".csv";
        // Export::saveVectorCSV(filename, amps);

        // filename = "FilesCSV/vector_b_" + std::to_string(i) + ".csv";
        // Export::saveVectorCSV(filename, b);

    }

    this->UPW_ = UPW_list;
    this->amplitudes_ext_ = amplitudes_ext;
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

Eigen::VectorXd FieldCalculatorTotal::computePower(
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
    std::complex<double> integral, integrand;
    Eigen::Vector3cd cross;

    int grid_size = sqrt(N);
    Eigen::MatrixXd integrand_mat(grid_size, grid_size);
    // int row, col;

    for (int j = 0; j < B; ++j){
        outE = Eigen::MatrixX3cd::Zero(N,3);
        outH = Eigen::MatrixX3cd::Zero(N,3);

        computeFields(outE, outH, points, j);
        integral = 0.0;  

        for (int i = 0; i < N; ++i){
            cross = 0.5 * outE.row(i).cross(outH.row(i).conjugate());
            integrand = cross.dot(normals.row(i));

            // if (j == 0){
            //     col = i % grid_size;
            //     row = i / grid_size;

            //     integrand_mat(row, col) = abs(integrand); 
            // }

            integral += integrand * dA;
        }

        integral_vec(j) = integral.real();

    }

    return integral_vec;   
}


std::pair<Eigen::VectorXd, Eigen::VectorXd> FieldCalculatorTotal::computeTangentialError(int polarization_index){
    auto control_points = mas_.getControlPoints();
    auto control_tangents1 = mas_.getControlTangents1();
    auto control_tangents2 = mas_.getControlTangents2();

    int N = control_points.rows();

    // Compute field from interior points
    Eigen::MatrixX3cd intE = Eigen::MatrixX3cd::Zero(N, 3);
    Eigen::MatrixX3cd intH = Eigen::MatrixX3cd::Zero(N, 3); 

    computeFields(intE, intH, control_points);

    // Compute field from exterior points and add UPW
    Eigen::MatrixX3cd extE = Eigen::MatrixX3cd::Zero(N, 3);
    Eigen::MatrixX3cd extH = Eigen::MatrixX3cd::Zero(N, 3); 
    Eigen::MatrixX3cd Ei(N, 3), Hi(N, 3);

    for (size_t i = 0; i < dipoles_ext_.size(); ++i) {
        dipoles_ext_[i]->computeFields(Ei, Hi, control_points);

        extE += amplitudes_ext_.row(polarization_index)(i) * Ei; 
        extH += amplitudes_ext_.row(polarization_index)(i) * Hi;
    }

    UPW_[polarization_index]->computeFields(Ei, Hi, control_points);
    extE += Ei;
    extH += Hi;


    // Check tangentiel elements
    Eigen::ArrayXd tangential_error1(N), tangential_error2(N);

    tangential_error1 = ((extE - intE).array() * control_tangents1.array()).rowwise().sum().abs();
    tangential_error2 = ((extE - intE).array() * control_tangents2.array()).rowwise().sum().abs();

    return {tangential_error1.matrix(), tangential_error2.matrix()};

}
