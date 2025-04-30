#include "FieldCalculatorUPW.h"
#include <complex>
#include <Eigen/Dense>
#include "UtilsTransform.h"
#include "UtilsFresnel.h"

using namespace Eigen;
using namespace std;

FieldCalculatorUPW::FieldCalculatorUPW(const Eigen::Vector3d& k_in, const double E0_in, const double polarization_in, 
    const Constants& constants_in)
: k_(k_in), E0_(E0_in), polarization_(polarization_in), constants_(constants_in) {}


void FieldCalculatorUPW::computeReflectedFields(
    Eigen::MatrixX3cd& refE,
    Eigen::MatrixX3cd& refH,
    const Eigen::MatrixX3d& evalPoints
) const {
    int N = evalPoints.rows();

    double cosPhi, sinPhi, cosTheta_in, sinTheta_in; // azimuthal angle the same for k_in and k_rot????

    TransformUtils::computeAngles(k_, 1.0, cosTheta_in, sinTheta_in,
        cosPhi, sinPhi);
    double cosBeta = cos(polarization_);
    double sinBeta = sin(polarization_);

    auto Rz = TransformUtils::rotationMatrixZ(cosPhi, sinPhi);
    auto Rz_inv = TransformUtils::rotationMatrixZInv(cosPhi, sinPhi);

    // Vector3d k_rot = Rz * k;

    complex<double> Gamma_r_perp = UtilsFresnel::fresnelTE(cosTheta_in, sinTheta_in, constants.epsilon0, constants.epsilon1).first;
    complex<double> Gamma_r_par = UtilsFresnel::fresnelTM(cosTheta_in, sinTheta_in, constants.epsilon0, constants.epsilon1).first;

    for (int i = 0; i < N; ++i) {
        Vector3d x = evalPoints.row(i);
        
        Eigen::Vector3d x_rot = Rz_inv * x;
        // Vector3d x_plane (x_rot[0], 0.0, x_rot[2]);

        complex<double> expk_rot = E0_ * exp(-constants.j * constants.k0 * (sinTheta_in * x_rot[0] - cosTheta_in * x_rot[2]));
    
        Eigen::Vector3cd E_perp(0.0, Gamma_r_perp * expk_rot, 0.0);
        Eigen::Vector3cd H_perp(cosTheta_in * Gamma_r_perp * expk_rot / constants.eta0, 0.0, sinTheta_in * Gamma_r_perp * expk_rot / constants.eta0);

        Eigen::Vector3cd E_par(-cosTheta_in * Gamma_r_par * expk_rot, 0.0, -sinTheta_in * Gamma_r_par * expk_rot);
        Eigen::Vector3cd H_par(0.0, - Gamma_r_perp * expk_rot/constants.eta0, 0);

        refE.row(i) = Rz * (cosBeta * E_perp + sinBeta * E_par);
        refH.row(i) = Rz * (cosBeta * H_perp + sinBeta * H_par);

    }
}



void FieldCalculatorUPW::computeFields(
    MatrixX3cd& outE,
    MatrixX3cd& outH,
    const MatrixX3d& evalPoints,
    int polarization_idx // Only used in total fields
) const {
    int N = evalPoints.rows();
    
    if (evalPoints.rows() == 0) {
        outE.resize(0, 3);
        outH.resize(0, 3);
        return;
    }
    

    double cosPhi, sinPhi, cosTheta_in, sinTheta_in; // azimuthal angle the same for k_in and k_rot????

    TransformUtils::computeAngles(-k_, 1.0, cosTheta_in, sinTheta_in,
        cosPhi, sinPhi);

    double cosBeta = cos(polarization_);
    double sinBeta = sin(polarization_);

    auto Rz = TransformUtils::rotationMatrixZ(cosPhi, sinPhi);
    auto Rz_inv = TransformUtils::rotationMatrixZInv(cosPhi, sinPhi);

    for (int i = 0; i < N; ++i) {
        Vector3d x = evalPoints.row(i);

        Eigen::Vector3d x_rot = Rz_inv * x;

        complex<double> phase1 = E0_ * exp(- constants.j * constants.k0 * (x_rot[0] * sinTheta_in - x_rot[2] * cosTheta_in));
        
        Vector3cd E_in_perp (0.0, phase1, 0.0);
        Vector3cd E_in_par (cosTheta_in * phase1, 0.0, sinTheta_in * phase1);
        Vector3cd E_in = Rz * (cosBeta * E_in_perp + sinBeta * E_in_par);


        Vector3cd H_in_perp (- cosTheta_in * phase1/constants.eta0, 0.0, - sinTheta_in * phase1/constants.eta0);
        Vector3cd H_in_par (0.0, phase1/constants.eta0, 0.0);
        Vector3cd H_in = Rz * (cosBeta * H_in_perp + sinBeta * H_in_par);

        outE.row(i) = E_in.transpose(); // + refE.row(i);
        outH.row(i) = H_in.transpose(); // + refH.row(i);
    }
}

