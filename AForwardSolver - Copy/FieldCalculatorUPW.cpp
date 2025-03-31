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

     

    Vector3cd H = (k.cross(E0)) / (constants.omega * constants.mu0) * phase;

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
