#include <Eigen/Dense>
#include <chrono>
#include "../../lib/Forward/SystemAssembler.h"
#include "../../lib/Forward/FieldCalculator.h"
#include "../../lib/Forward/FieldCalculatorTotal.h"
#include "../../lib/Forward/FieldCalculatorDipole.h"
#include "../../lib/Forward/FieldCalculatorUPW.h"
#include "../../lib/Forward/FieldCalculatorArray.h"
#include "../../lib/Utils/Constants.h"
#include "../../lib/Utils/UtilsDipole.h"
#include "../../lib/Utils/UtilsSolvers.h"
#include "../../lib/Utils/UtilsExport.h"
#include "../../lib/Utils/UtilsTransform.h"


FieldCalculatorArray::FieldCalculatorArray(
    const MASSystem& masSystem, const double largeDim, bool verbose
) : masSystem_(masSystem), FieldTotal_(std::make_shared<FieldCalculatorTotal>(masSystem, verbose)), 
    largeDim_(largeDim)
{
    // Maybe some constructor for the weights?
    constructor();

    std::cout << std::endl;
    std::cout << "Array calculator initialized!" << std::endl << std::endl;
}

void FieldCalculatorArray::constructor()
{
    auto points = masSystem_.getPoints();
    double dimension = points.col(0).maxCoeff() - points.col(0).minCoeff();
    double half_dim = dimension/2.0;

    int N = std::ceil(largeDim_/dimension);

    this->arrayWeights_ = Eigen::VectorXd::Constant(N * N, 1.0);

    Eigen::MatrixX3d arrayPositions(N * N, 3);

    Eigen::Vector3d v;

    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            
            arrayPositions.row(N * i + j) << -half_dim + i * 1.0/N * dimension, 
                                             -half_dim + j * 1.0/N * dimension, 
                                              0;

        };
    };

    this->arrayPositions_ = arrayPositions;

}


void FieldCalculatorArray::computeFields(
    Eigen::MatrixX3cd& outE,
    Eigen::MatrixX3cd& outH,
    const Eigen::MatrixX3d& evalPoints,
    int polarization_idx
) const {
    std::cout << "Near field not implemented for array equations, calculating far field." << std::endl;
    computeFarFields(outE, outH, evalPoints, polarization_idx);
}


void FieldCalculatorArray::computeFarFields(
    Eigen::MatrixX3cd& outE,
    Eigen::MatrixX3cd& outH,
    const Eigen::MatrixX3d& evalPoints,
    int polarization_idx
) const {
    int N = arrayPositions_.rows();
    int M = evalPoints.rows();

    FieldTotal_->computeFields(outE, outH, evalPoints, polarization_idx);
    Eigen::Vector3d x;

    for (int j = 0; j < M; ++j){
        x = evalPoints.row(j).normalized();

        std::complex<double> arrayFactor = 0.0;

        for (int i = 0; i < N; i++){
            arrayFactor += arrayWeights_(i) * exp(-constants.j * constants.k0 * arrayPositions_.row(i).dot(x));
        };

        outE.row(j) *= arrayFactor;
        outH.row(j) *= arrayFactor;
    };

}



Eigen::VectorXd FieldCalculatorArray::computePower(
    const Surface& surface
){
    Eigen::MatrixX3d points = surface.getPoints();
    Eigen::MatrixX3d normals = surface.getNormals();
    int N = points.rows();

    int B = masSystem_.getPolarizations().rows();

    // We assume uniform grid distances
    double dx = (points.row(1) - points.row(0)).norm();
    double dA = dx * dx;
    
    Eigen::MatrixX3cd outE, outH;
    Eigen::VectorXd integral_vec(B);
    std::complex<double> integral, integrand;
    Eigen::Vector3cd cross;

    for (int j = 0; j < B; ++j){
        outE = Eigen::MatrixX3cd::Zero(N,3);
        outH = Eigen::MatrixX3cd::Zero(N,3);

        computeFarFields(outE, outH, points, j);
        integral = 0.0;  

        for (int i = 0; i < N; ++i){
            cross = outE.row(i).cross(outH.row(i).conjugate());
            integrand = cross.dot(normals.row(i));
            integral += integrand;

            // if (i < 5) {
            //     std::cout << "E[" << i << "]: " << outE.row(i) << "\n";
            //     std::cout << "H[" << i << "]: " << outH.row(i) << "\n";
            //     std::cout << "cross[" << i << "]: " << cross.transpose() << "\n";
            //     std::cout << "normal[" << i << "]: " << normals.row(i) << "\n";
            //     std::cout << "integrand[" << i << "]: " << integrand << "\n\n";
            // };
        }

        integral_vec(j) = 0.5 * integral.real() * dA;

    }

    return integral_vec;   

}

