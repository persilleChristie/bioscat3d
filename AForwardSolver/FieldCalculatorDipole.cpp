#include "FieldCalculatorDipole.h"
#include <cmath>
#include <complex>
#include "Constants.h"

using namespace Eigen;
using namespace std;

static const complex<double> j(0.0, 1.0);

FieldCalculatorDipole::FieldCalculatorDipole(const Dipole& dipole)
    : dip(dipole) {}

void FieldCalculatorDipole::computeFields(
    Eigen::MatrixXd& outE,
    Eigen::MatrixXd& outH,
    const Eigen::MatrixXd& evalPoints
) const {
    int N = evalPoints.rows();
    outE.resize(N, 3);
    outH.resize(N, 3);

    const auto& xPrime = dip.getPosition();
    const auto& rPrime = dip.getMoment();
    double Iel = 1.0;  // Placeholder — should be passed or part of dipole later

    for (int i = 0; i < N; ++i) {
        Vector3d x = evalPoints.row(i);
        Vector3d x_origo = x - xPrime;
        double r = x_origo.norm();

        double cos_theta = x_origo[2] / r;
        double sin_theta = sqrt(1 - cos_theta * cos_theta);
        double xy_norm = hypot(x_origo[0], x_origo[1]);

        double cos_phi = (xy_norm == 0) ? 1.0 : x_origo[0] / xy_norm;
        double sin_phi = (xy_norm == 0) ? 0.0 : x_origo[1] / xy_norm;

        complex<double> expK0r = polar(1.0, -constants.k0 * r);

        // E-field in spherical components
        complex<double> E_r = constants.eta0 * Iel * cos_theta / (2.0 * constants.pi * r * r) *
                              (1.0 + 1.0 / (j * constants.k0 * r)) * expK0r;
        complex<double> E_theta = (j * constants.eta0 * Iel * sin_theta / (4.0 * constants.pi * r)) *
                                  (1.0 + 1.0 / (j * constants.k0 * r) - 1.0 / (constants.k0 * r * r)) * expK0r;

        Vector3cd E_cartesian = {
            E_r * sin_theta * cos_phi + E_theta * cos_theta * cos_phi,
            E_r * sin_theta * sin_phi + E_theta * cos_theta * sin_phi,
            E_r * cos_theta - E_theta * sin_theta
        };

        complex<double> H_phi = j * constants.k0 * Iel * sin_theta / (4.0 * constants.pi * r) *
                                (1.0 + 1.0 / (j * constants.k0 * r)) * expK0r;

        Vector3cd H_cartesian = {
            -H_phi * sin_phi,
            H_phi * cos_phi,
            0.0
        };

        // Rotate and translate
        double cosT = cos_theta;
        double sinT = sin_theta;
        Matrix3d Ry {{ cosT, 0, sinT }, { 0, 1, 0 }, { -sinT, 0, cosT }};

        Matrix3d Rz {{ cos_phi, -sin_phi, 0 }, { sin_phi, cos_phi, 0 }, { 0, 0, 1 }};

        Vector3cd E_rot = Rz * Ry * E_cartesian;
        Vector3cd H_rot = Rz * Ry * H_cartesian;

        Vector3cd E_final = E_rot - xPrime;
        Vector3cd H_final = H_rot - xPrime;

        outE.row(i) = E_final.real().transpose();
        outH.row(i) = H_final.real().transpose();
    }
}
