<<<<<<< HEAD
#include "FieldCalculatorUPW.h"
#include <complex>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

const complex<double> j(0, 1.0);  // Imaginary unit

// Compute full field at one point (used internally)
pair<Vector3cd, Vector3cd> computeUPWField(const Vector3d& x, const Vector3d& k, const Vector3cd& E0, const Constants& constants) {
    complex<double> phase = exp(-j * k.dot(x));
    Vector3cd E = E0 * phase;

    double omega = 1; // constants.omega; ?????????????????????????????????????????????
    double mu0 = constants.mu0;

    Vector3cd H = (k.cross(E0)) / (omega * mu0) * phase;

    return {E, H};
}

void FieldCalculatorUPW::computeFields(
    MatrixXd& outE_real,
    MatrixXd& outH_real,
    const MatrixXd& evalPoints
) const {
    int N = evalPoints.rows();
    outE_real.resize(N, 3);
    outH_real.resize(N, 3);

    for (int i = 0; i < N; ++i) {
        Vector3d x = evalPoints.row(i);
        auto [E, H] = computeUPWField(x, k, E0, constants);
        outE_real.row(i) = E.real();
        outH_real.row(i) = H.real();
    }
}

Vector3cd FieldCalculatorUPW::getEField(const Vector3d& x) const {
    auto [E, _] = computeUPWField(x, k, E0, constants);
    return E;
}

Vector3cd FieldCalculatorUPW::getHField(const Vector3d& x) const {
    auto [_, H] = computeUPWField(x, k, E0, constants);
    return H;
}
=======
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
>>>>>>> a5573bfeda0575951aeed785824f8fe3a44f17d7
