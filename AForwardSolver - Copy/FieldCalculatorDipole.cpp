#include "FieldCalculatorDipole.h"
#include "UtilsTransform.h"
#include <complex>

using namespace Eigen;
using namespace std;
using namespace TransformUtils;

FieldCalculatorDipole::FieldCalculatorDipole(const Dipole& dipole, double Iel, const Constants& constants) // drop Iel and constants, add k0 
    : dipole_(dipole), Iel_(Iel), constants_(constants) {}

std::pair<Vector3cd, Vector3cd> FieldCalculatorDipole::computeFieldAt(const Vector3d& x) const {
    // Compute dipole direction angles
    double cosTheta, sinTheta, cosPhi, sinPhi;
    computeAngles(dipole_.getDirection(), 1.0, cosTheta, sinTheta, cosPhi, sinPhi);

    // Rotate to local dipole coordinates
    auto Ry = rotationMatrixY(cosTheta, sinTheta);
    auto Rz = rotationMatrixZ(cosPhi, sinPhi);

    Vector3d x_local = Rz * Ry * (x - dipole_.getPosition());
    double r = x_local.norm();

    if (r == 0.0) {
        return {Vector3cd::Zero(), Vector3cd::Zero()};
    }

    complex<double> expK0r = exp(-constants.j * constants_.k0 * r);

    complex<double> E_r = constants_.eta0 * Iel_ * cosTheta / (2.0 * constants_.pi * r * r)
                        * (1.0 + 1.0 / (constants.j * constants_.k0 * r)) * expK0r;

    complex<double> E_theta = (constants.j * constants_.eta0 * Iel_ * sinTheta / (4.0 * constants_.pi * r))
                            * (1.0 + 1.0 / (constants.j * constants_.k0 * r) - 1.0 / (constants_.k0 * r * r)) * expK0r;

    Vector3cd E_local = computeECartesian(sinTheta, cosTheta, sinPhi, cosPhi, E_r, E_theta);

    complex<double> H_phi = constants.j * constants_.k0 * Iel_ * sinTheta / (4.0 * constants_.pi * r)
                          * (1.0 + 1.0 / (constants.j * constants_.k0 * r)) * expK0r;

    Vector3cd H_local;
    H_local << -H_phi * sinPhi, H_phi * cosPhi, 0.0;

    // Rotate back to global coordinates
    auto Ry_inv = rotationMatrixYInv(cosTheta, sinTheta);
    auto Rz_inv = rotationMatrixZInv(cosPhi, sinPhi);
    Matrix3d R = Ry_inv * Rz_inv;

    Vector3cd E_global = R.cast<complex<double>>() * E_local;
    Vector3cd H_global = R.cast<complex<double>>() * H_local;

    return {E_global, H_global};
}

Vector3cd FieldCalculatorDipole::getEField(const Vector3d& x) const {
    return computeFieldAt(x).first;
}

Vector3cd FieldCalculatorDipole::getHField(const Vector3d& x) const {
    return computeFieldAt(x).second;
}
