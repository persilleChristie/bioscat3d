#ifndef FIELDCALCULATORUPW_CPP
#define FIELDCALCULATORUPW_CPP

#include "FieldCalculatorUPW.h"
#include "Constants.h"
#include <cmath>
#include <complex>

using namespace Eigen;
using namespace std;

FieldCalculatorUPW::FieldCalculatorUPW(const Eigen::Vector3d& waveVector,
                                       const Eigen::Vector3d& polarization)
    : k(waveVector), E0(polarization) {
    if (abs(k.dot(E0)) > 1e-10) {
        throw std::invalid_argument("Polarization vector must be perpendicular to wave vector.");
    }
}

void FieldCalculatorUPW::computeFields(
    Eigen::MatrixXd& outE,
    Eigen::MatrixXd& outH,
    const Eigen::MatrixXd& evalPoints
) const {
    int N = evalPoints.rows();
    outE.resize(N, 3);
    outH.resize(N, 3);

    double k_norm = k.norm();
    Eigen::Vector3d k_hat = k / k_norm;
    Eigen::Vector3d H0_real = (k_hat.cross(E0)) / (constants.mu0); /// LOOK AT ME!

    
    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d& x = evalPoints.row(i);
        double phase = k.dot(x);
        complex<double> phasor = exp(-constants.j * phase);

        Eigen::Vector3cd E = phasor * E0;
        Eigen::Vector3cd H = phasor * H0_real;

        outE.row(i) = E.real().transpose();
        outH.row(i) = H.real().transpose();
    }
}

#endif // FIELDCALCULATORUPW_CPP