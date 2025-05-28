#include <Eigen/Dense>
#include "../../lib/Forward/FieldCalculatorUPW.h"
#include "../../lib/Utils/UtilsTransform.h"
#include "../../lib/Utils/Constants.h"


FieldCalculatorUPW::FieldCalculatorUPW(const Eigen::Vector3d& k_in, const double E0_in, const double polarization_in)
: k_(k_in), E0_(E0_in), polarization_(polarization_in) {}



void FieldCalculatorUPW::computeFields(
    Eigen::MatrixX3cd& outE,
    Eigen::MatrixX3cd& outH,
    const Eigen::MatrixX3d& evalPoints,
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

    // print angles
    std::cout << "cosTheta_in: " << cosTheta_in << ", sinTheta_in: " << sinTheta_in << std::endl;
    std::cout << "cosPhi: " << cosPhi << ", sinPhi: " << sinPhi << std::endl;
    std::cout << "cosBeta: " << cosBeta << ", sinBeta: " << sinBeta << std::endl;
    std::cout << "k_: " << k_.transpose() << std::endl;
    std::cout << "E0_: " << E0_ << std::endl;
    std::cout << "polarization_: " << polarization_ << std::endl;

    auto Rz = TransformUtils::rotationMatrixZ(cosPhi, sinPhi);
    auto Rz_inv = TransformUtils::rotationMatrixZInv(cosPhi, sinPhi);

    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d x = evalPoints.row(i);

        Eigen::Vector3d x_rot = Rz_inv * x;

        std::complex<double> phase1 = E0_ * exp(- constants.j * constants.k0 * (x_rot[0] * sinTheta_in - x_rot[2] * cosTheta_in));
        // print constants.k0
        std::cout << "constants.k0: " << constants.k0 << std::endl;
        // print x_rot
        std::cout << "x_rot: " << x_rot.transpose() << std::endl;
        // print phase1
        std::cout << "Phase1 for point " << i << ": " << phase1 << std::endl;
        Eigen::Vector3cd E_in_perp (0.0, phase1, 0.0);
        Eigen::Vector3cd E_in_par (- cosTheta_in * phase1, 0.0, - sinTheta_in * phase1);
        Eigen::Vector3cd E_in = Rz * (cosBeta * E_in_perp + sinBeta * E_in_par);


        Eigen::Vector3cd H_in_perp (cosTheta_in * phase1/constants.eta0, 0.0, sinTheta_in * phase1/constants.eta0);
        Eigen::Vector3cd H_in_par (0.0, phase1/constants.eta0, 0.0);
        Eigen::Vector3cd H_in = Rz * (cosBeta * H_in_perp + sinBeta * H_in_par);
        

        outE.row(i) = E_in.transpose(); 
        outH.row(i) = H_in.transpose(); 
    }
}

