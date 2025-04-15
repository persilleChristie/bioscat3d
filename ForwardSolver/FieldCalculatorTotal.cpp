#include "FieldCalculatorTotal.h"
#include <cmath>


FieldCalculatorTotal::FieldCalculatorTotal(
    const Eigen::VectorXcd & amplitudes,
    const std::vector<std::shared_ptr<FieldCalculatorDipole>>& dipoles,
    const std::shared_ptr<FieldCalculatorUPW>& UPW
) 
    : amplitudes_(amplitudes), dipoles_(dipoles), UPW_(UPW)
{
    if (amplitudes_.size() != dipoles_.size()) {
        throw std::invalid_argument("Amplitudes and dipoles must have the same size.");
    }
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

    Eigen::MatrixX3cd outE(N,3), outH(N,3);
    
    computeFields(outE, outH, points);

    std::complex<double> integrand, integral = 0.0;

    for (int i = 0; i < N; ++i){
        integrand = (1/2 * outE.row(i).cross(outH.row(i).conjugate())).dot(normals.row(i));

        integral += integrand * dA;
    }
}
