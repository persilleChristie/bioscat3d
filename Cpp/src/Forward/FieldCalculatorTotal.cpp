#include <Eigen/Dense>
#include <chrono>
#include "../../lib/Forward/SystemAssembler.h"
#include "../../lib/Forward/FieldCalculator.h"
#include "../../lib/Forward/FieldCalculatorTotal.h"
#include "../../lib/Forward/FieldCalculatorDipole.h"
#include "../../lib/Forward/FieldCalculatorUPW.h"
#include "../../lib/Utils/Constants.h"
#include "../../lib/Utils/UtilsDipole.h"
#include "../../lib/Utils/UtilsSolvers.h"
#include "../../lib/Utils/UtilsExport.h"

// MassSystem is copied here?
FieldCalculatorTotal::FieldCalculatorTotal(
    const MASSystem& masSystem, bool verbose
) : mas_(masSystem)
{
    constructor(verbose);
}

void FieldCalculatorTotal::constructor(bool verbose)
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
    // this->dipoles_ = std::move(sources_int);
    this->dipoles_ = sources_int;
    this->dipoles_ext_ = sources_ext;

    // -------- SOLVE SYSTEM ---------
    // Allocate space for system matrix and UPW
    Eigen::MatrixXcd A(4 * M, 2 * N); 
    Eigen::VectorXcd b(4 * M);
    std::shared_ptr<FieldCalculatorUPW> UPW;
    std::vector<std::shared_ptr<FieldCalculator>> UPW_list;

    // Load polarizations and incidence vector
    // const auto& [kinc, lambda] = mas_.getInc();
    auto kinc = mas_.getInc();
    double lambda = constants.getWavelength();
    Eigen::VectorXd polarizations = mas_.getPolarizations();

    int B = polarizations.size();

    // Allocate space for amplitudes
    Eigen::MatrixXcd amplitudes(B, N);
    Eigen::MatrixXcd amplitudes_ext(B, N);
    

    for (int i = 0; i < B; ++i){
        if (verbose)
        {std::cout << "----------------- Beta nr. " << i + 1 << "/" << B << " -----------------" << std::endl;}
        UPW = std::make_shared<FieldCalculatorUPW>(kinc, 1.0, polarizations(i));
        // UPW_list.push_back(UPW) 
        UPW_list.emplace_back(UPW);

        // ----------------- Assemble system and time --------------------
        if (verbose) {
            std::cout << "Assembling system..." << std::endl;
        };

        auto start_assemble = std::chrono::high_resolution_clock::now();

        SystemAssembler::assembleSystem(A, b, points, t1, t2, sources_int, sources_ext, UPW);

        auto stop_assemble = std::chrono::high_resolution_clock::now();

        auto duration_assemble = std::chrono::duration_cast<std::chrono::seconds>(stop_assemble - start_assemble);

        if (verbose){
            std::cout << "System assembled in " << duration_assemble.count() << " seconds" << std::endl;
        }


        // ----------------- Solve system and time --------------------
        if (verbose){
            std::cout << "Solving system..." << std::endl;
        };

        auto start_solve = std::chrono::high_resolution_clock::now();

        auto amps = UtilsSolvers::solveQR(A, b);

        auto stop_solve = std::chrono::high_resolution_clock::now();

        auto duration_solve = std::chrono::duration_cast<std::chrono::seconds>(stop_solve - start_solve);

        if (verbose){
        std::cout << "System solved in " << duration_solve.count() << " seconds" << std::endl;
        }
        
        amplitudes.row(i) = amps.head(N);
        amplitudes_ext.row(i) = amps.tail(N);

    }

    this->UPW_ = UPW_list;
    this->amplitudes_ext_ = amplitudes_ext;
    this->amplitudes_ = amplitudes;

    if (verbose)
    {
        std::cout << std::endl << "Field Calculator initialized successfully!" << std::endl;
    };
}


void computeLinearCombinations(Eigen::MatrixX3cd& outE,
    Eigen::MatrixX3cd& outH,
    const Eigen::MatrixX3d& evalPoints,
    int polarization_idx,
    std::vector<std::shared_ptr<FieldCalculator>> dipoles,
    const Eigen::MatrixXcd& amplitudes, 
    bool nearField
    ){

    int M = evalPoints.rows();

    Eigen::MatrixX3cd Ei(M, 3), Hi(M, 3);

    int N = static_cast<int>(dipoles.size()) / 2;

    if (nearField){
        for (int i = 0; i < N; ++i) {
            dipoles[2 * i]->computeFields(Ei, Hi, evalPoints);

            outE += amplitudes.row(polarization_idx)(i) * Ei;
            outH += amplitudes.row(polarization_idx)(i) * Hi;

            dipoles[2 * i + 1]->computeFields(Ei, Hi, evalPoints);

            outE += amplitudes.row(polarization_idx)(i + N) * Ei;
            outH += amplitudes.row(polarization_idx)(i + N) * Hi;
        };
    } else {
       for (int i = 0; i < N; ++i) {
            dipoles[2 * i]->computeFarFields(Ei, Hi, evalPoints);

            outE += amplitudes.row(polarization_idx)(i) * Ei;
            outH += amplitudes.row(polarization_idx)(i) * Hi;

            dipoles[2 * i + 1]->computeFarFields(Ei, Hi, evalPoints);

            outE += amplitudes.row(polarization_idx)(i + N) * Ei;
            outH += amplitudes.row(polarization_idx)(i + N) * Hi;
        };
    }
}


void FieldCalculatorTotal::computeFields(
    Eigen::MatrixX3cd& outE,
    Eigen::MatrixX3cd& outH,
    const Eigen::MatrixX3d& evalPoints,
    int polarization_idx
) const {
    computeLinearCombinations(outE, outH, evalPoints, polarization_idx, dipoles_, amplitudes_, true);
}


void FieldCalculatorTotal::computeFarFields(
    Eigen::MatrixX3cd& outE,
    Eigen::MatrixX3cd& outH,
    const Eigen::MatrixX3d& evalPoints,
    int polarization_idx
) const {
    computeLinearCombinations(outE, outH, evalPoints, polarization_idx, dipoles_, amplitudes_, false);
}



Eigen::VectorXd FieldCalculatorTotal::computePower(
    const Surface& surface
) const {
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

        computeFarFields(outE, outH, points, j);
        integral = 0.0;  

        for (int i = 0; i < N; ++i){
            cross = outE.row(i).cross(outH.row(i).conjugate());
            integrand = cross.dot(normals.row(i));
            integral += integrand;
        }

        integral_vec(j) = 0.5 * integral.real() * dA;

    }

    return integral_vec;   
}


std::vector<Eigen::VectorXd> FieldCalculatorTotal::computeTangentialError(int polarization_index){
    auto control_points = mas_.getControlPoints();
    auto control_tangents1 = mas_.getControlTangents1();
    auto control_tangents2 = mas_.getControlTangents2();

    int N = control_points.rows();

    // Compute field from interior points
    Eigen::MatrixX3cd intE = Eigen::MatrixX3cd::Zero(N, 3);
    Eigen::MatrixX3cd intH = Eigen::MatrixX3cd::Zero(N, 3); 

    computeLinearCombinations(intE, intH, control_points, polarization_index, dipoles_, amplitudes_, true);

    // Compute field from exterior points 
    Eigen::MatrixX3cd extE = Eigen::MatrixX3cd::Zero(N, 3);
    Eigen::MatrixX3cd extH = Eigen::MatrixX3cd::Zero(N, 3); 

    computeLinearCombinations(extE, extH, control_points, polarization_index, dipoles_ext_, amplitudes_ext_, true);

    // Compute incident field
    Eigen::MatrixX3cd incE(N, 3), incH(N, 3);
    UPW_[polarization_index]->computeFields(incE, incH, control_points);

    // Calculate tangentiel elements
    Eigen::ArrayXd E_tangential_error1(N), E_tangential_error2(N), H_tangential_error1(N), H_tangential_error2(N);

    Eigen::ArrayX3cd E_diff = (intE - extE + incE).array();
    Eigen::ArrayX3cd H_diff = (intH - extH + incH).array();

    E_tangential_error1 = (E_diff * control_tangents1.array()).rowwise().sum().abs();
    E_tangential_error2 = (E_diff * control_tangents2.array()).rowwise().sum().abs();
    H_tangential_error1 = (H_diff * control_tangents1.array()).rowwise().sum().abs();
    H_tangential_error2 = (H_diff * control_tangents2.array()).rowwise().sum().abs();

    return {E_tangential_error1.matrix(), E_tangential_error2.matrix(), H_tangential_error1.matrix(), H_tangential_error2.matrix()};
}
