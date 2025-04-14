#include "FieldCalculatorUPW.h"
#include <complex>
#include <Eigen/Dense>
#include "UtilsTransform.h"
#include "UtilsFresnel.h"

using namespace Eigen;
using namespace std;

FieldCalculatorUPW::FieldCalculatorUPW(const Eigen::Vector3d& k_in, const double E0_in, const double polarization_in, 
    const Constants& constants_in)
: k(k_in), E0(E0_in), polarization(polarization_in), constants(constants_in) {}


void FieldCalculatorUPW::computeReflectedFields(
    Eigen::MatrixX3cd& refE,
    Eigen::MatrixX3cd& refH,
    const Eigen::MatrixX3d& evalPoints
) const {
    int N = evalPoints.rows();

    double cosPhi, sinPhi, cosTheta_in, sinTheta_in; // azimuthal angle the same for k_in and k_rot????

    TransformUtils::computeAngles(k, 1.0, cosTheta_in, sinTheta_in,
        cosPhi, sinPhi);

    cout << "cos(theta) = " << cosTheta_in << endl;

    double cosBeta = cos(polarization);
    double sinBeta = sin(polarization);

    auto Rz = TransformUtils::rotationMatrixZ(cosPhi, sinPhi);
    auto Rz_inv = TransformUtils::rotationMatrixZInv(cosPhi, sinPhi);

    // Vector3d k_rot = Rz * k;

    complex<double> Gamma_r_perp = UtilsFresnel::fresnelTE(cosTheta_in, sinTheta_in, constants.epsilon0, constants.epsilon1).first;
    complex<double> Gamma_r_par = UtilsFresnel::fresnelTM(cosTheta_in, sinTheta_in, constants.epsilon0, constants.epsilon1).first;

    if (Gamma_r_perp.imag() == 0){
    cout << "Gamma perp: " << Gamma_r_perp.real() << endl;}
    else {cout << "Gamma perp: " << Gamma_r_perp.real() << "+" << Gamma_r_perp.imag() << "j" << endl;}
    if (Gamma_r_par.imag() == 0){
        cout << "Gamma par: " << Gamma_r_par.real() << endl;}
    else {cout << "Gamma perp: " << Gamma_r_par.real() << "+" << Gamma_r_par.imag() << "j" << endl;}


    for (int i = 0; i < N; ++i) {
        Vector3d x = evalPoints.row(i);
        
        Eigen::Vector3d x_rot = Rz * x;
        // Vector3d x_plane (x_rot[0], 0.0, x_rot[2]);

        complex<double> expk_rot = E0 * exp(-constants.j * constants.k0 * (sinTheta_in * x_rot[0] - cosTheta_in * x_rot[2]));
    
        Eigen::Vector3cd E_perp(0.0, Gamma_r_perp * expk_rot, 0.0);
        Eigen::Vector3cd H_perp(cosTheta_in * Gamma_r_perp * expk_rot / constants.eta0, 0.0, sinTheta_in * Gamma_r_perp * expk_rot / constants.eta0);

        Eigen::Vector3cd E_par(cosTheta_in * Gamma_r_par * expk_rot, 0.0, sinTheta_in * Gamma_r_par * expk_rot);
        Eigen::Vector3cd H_par(0.0, - Gamma_r_perp * expk_rot/constants.eta0, 0);

        refE.row(i) = Rz_inv * (cosBeta * E_perp + sinBeta * E_par);
        refH.row(i) = Rz_inv * (cosBeta * H_perp + sinBeta * H_par);

    }
}



void FieldCalculatorUPW::computeFields(
    MatrixX3cd& outE,
    MatrixX3cd& outH,
    const MatrixX3d& evalPoints
) const {
    int N = evalPoints.rows();
    
    if (evalPoints.rows() == 0) {
        outE.resize(0, 3);
        outH.resize(0, 3);
        return;
    }
    

    double cosPhi, sinPhi, cosTheta_in, sinTheta_in; // azimuthal angle the same for k_in and k_rot????

    TransformUtils::computeAngles(-k, 1.0, cosTheta_in, sinTheta_in,
        cosPhi, sinPhi);

    std::cout << "Incident angle: " << std::endl;
    std::cout << "cos(Theta) = " << cosTheta_in << endl;
    cout << "sin(Theta) = " << sinTheta_in << endl;

    double cosBeta = cos(polarization);
    double sinBeta = sin(polarization);

    auto Rz = TransformUtils::rotationMatrixZ(cosPhi, sinPhi);
    auto Rz_inv = TransformUtils::rotationMatrixZInv(cosPhi, sinPhi);
    
    cout << "Rotation matrix: " << endl;
    cout << Rz << endl;

    for (int i = 0; i < N; ++i) {
        Vector3d x = evalPoints.row(i);

        Eigen::Vector3d x_rot = Rz * x;
        
        cout << "Point of evaluation: " << x_rot << endl;

        complex<double> phase1 = E0 * exp(- constants.j * constants.k0 * (x_rot[0] * sinTheta_in - x_rot[2] * cosTheta_in));
        
        Vector3cd E_in_perp (0.0, phase1, 0.0);
        Vector3cd E_in_par (cosTheta_in * phase1, 0.0, sinTheta_in * phase1);
        Vector3cd E_in = Rz_inv * (cosBeta * E_in_perp + sinBeta * E_in_par);

        cout << "E_in_perp = " << E_in_perp << endl;
        cout << "E_in_par = " << E_in_par << endl;
        cout << "Calculated E_incident: " << E_in << endl;


        Vector3cd H_in_perp (- cosTheta_in * phase1/constants.eta0, 0.0, - sinTheta_in * phase1/constants.eta0);
        Vector3cd H_in_par (0.0, phase1/constants.eta0, 0.0);
        Vector3cd H_in = Rz_inv * (cosBeta * H_in_perp + sinBeta * H_in_par);

        cout << "H_in_perp = " << H_in_perp << endl;
        cout << "H_in_par = " << H_in_par << endl;
        cout << "Calculated H_incident: " << H_in << endl;


        outE.row(i) = E_in.transpose(); // + refE.row(i);
        outH.row(i) = H_in.transpose(); // + refH.row(i);
    }
}

