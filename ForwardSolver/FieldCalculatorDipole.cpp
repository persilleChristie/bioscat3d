#include "FieldCalculatorDipole.h"
#include "UtilsTransform.h"
#include <complex>


FieldCalculatorDipole::FieldCalculatorDipole(const Dipole& dipole, const Constants& constants) // drop Iel and constants, add k0 
    : dipole_(dipole), constants_(constants) {}

void FieldCalculatorDipole::computeFields(
    Eigen::MatrixX3cd& outE,
    Eigen::MatrixX3cd& outH,
    const Eigen::MatrixX3d& evalPoints
) const {

    int N = evalPoints.rows();
    if (N == 0) {
        outE.resize(0, 3);
        outH.resize(0, 3);
        return;
    }

    outE.resize(N, 3);
    outH.resize(N, 3);

    // Compute dipole direction angles
    double cosTheta, sinTheta, cosPhi, sinPhi;
    TransformUtils::computeAngles(dipole_.getDirection(), 1.0, cosTheta, sinTheta, cosPhi, sinPhi);

    // Rotate to local dipole coordinates
    auto Ry = TransformUtils::rotationMatrixY(cosTheta, sinTheta);
    auto Rz = TransformUtils::rotationMatrixZ(cosPhi, sinPhi);
    Eigen::Matrix3d R = Rz * Ry;

    // Rotate back to global coordinates
    auto Ry_inv = TransformUtils::rotationMatrixYInv(cosTheta, sinTheta);
    auto Rz_inv = TransformUtils::rotationMatrixZInv(cosPhi, sinPhi);
    Eigen::Matrix3d R_inverse = Ry_inv * Rz_inv;

    // Reserve memory for later
    double cosThetaX, sinThetaX, cosPhiX, sinPhiX;

    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d& x = evalPoints.row(i);

        Eigen::Vector3d x_local = R_inverse * (x - dipole_.getPosition());
        double r = x_local.norm();
        

        // Compute polar coordinates
        TransformUtils::computeAngles(x_local, r, cosThetaX, sinThetaX, cosPhiX, sinPhiX);

        std::complex<double> expK0r = std::exp(-constants_.j * constants_.k0 * r);

        if (r == 0.0) {
            outE.row(i) = Eigen::Vector3d::Zero();
            outH.row(i) = Eigen::Vector3d::Zero();
            continue;
        }

        std::complex<double> E_r = (constants_.eta0 * constants_.Iel * cosThetaX / (2.0 * constants_.pi * r * r)
                            * (1.0 + 1.0 / (constants_.j * constants_.k0 * r)) * expK0r);

        std::complex<double> E_theta = (constants_.j * constants_.eta0 * constants_.k0 * constants_.Iel * sinThetaX / (4.0 * constants_.pi * r)
                                * (1.0 + 1.0 / (constants_.j * constants_.k0 * r) - 1.0 / (constants_.k0 * constants_.k0 * r * r)) * expK0r);

        Eigen::Vector3cd E_local = TransformUtils::computeECartesian(sinThetaX, cosThetaX, sinPhiX, cosPhiX, E_r, E_theta);

        std::complex<double> H_phi = constants_.j * constants_.k0 * constants_.Iel * sinThetaX / (4.0 * constants_.pi * r)
                            * (1.0 + 1.0 / (constants_.j * constants_.k0 * r)) * expK0r;

        Eigen::Vector3cd H_local (-H_phi * sinPhiX, H_phi * cosPhiX, 0.0);

        outE.row(i) = R.cast<std::complex<double>>() * E_local;
        outH.row(i) = R.cast<std::complex<double>>() * H_local;
    };
}

