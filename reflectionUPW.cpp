#include <iostream>
#include <cmath>
#include <complex>
#include <Eigen/Dense>

using namespace std;

const complex<double> j(0, 1);  // Imaginary unit

// =========================================
//  Constants Definition
// =========================================

const double omega = 1.0;
const double epsilon0 = 1.0;
const double n0 = 1.0;
const double mu0 = 1.0;
const double k0 = omega * sqrt(epsilon0 * mu0);


const double epsilon_air = epsilon0;
const double epsilon_substrate = 2.0;
const double n_air = n0;
const double n_substrate = 2.0;
const double k_air = n_air * k0;
const double k_substrate = n_substrate * k0;

// =========================================
//  Azimuthal Angle Computation
// =========================================

inline double cos_phi(const Eigen::Vector3d& k_inc) {
    double xy_norm = hypot(k_inc[0], k_inc[1]);
    return (xy_norm == 0) ? 1.0 : k_inc[0] / xy_norm;
}

inline double sin_phi(const Eigen::Vector3d& k_inc) {
    double xy_norm = hypot(k_inc[0], k_inc[1]);
    return (xy_norm == 0) ? 0.0 : k_inc[1] / xy_norm;
}

// =========================================
//  Rotation Matrix (Z-axis)
// =========================================

Eigen::Matrix3d rotation_matrix_z(double cos_phi, double sin_phi) {
    return {
        Eigen::Matrix3d{
        {cos_phi, -sin_phi, 0},
        {sin_phi, cos_phi,  0},
        {0,       0,        1}
    }};
}

Eigen::Matrix3d rotation_matrix_z_inv(double cos_phi, double sin_phi) {
    return {
        Eigen::Matrix3d{
        {cos_phi, sin_phi, 0},
        {-sin_phi, cos_phi,  0},
        {0,       0,        1}
    }};
}


// =========================================
//  Polar Angle Computation
// =========================================

inline double cos_theta(const Eigen::Vector3d& k_rot) {
    return k_rot[2] / hypot(k_rot[0], k_rot[2]);
}

inline double sin_theta(const Eigen::Vector3d& k_rot) {
    double cosT = cos_theta(k_rot);
    return sqrt(1 - cosT * cosT);
}

// =========================================
//  Fresnel Coefficients (TE and TM)
// =========================================

pair<double, double> fresnel_coeffs_TE(double cos_theta_inc, double sin_theta_inc, double epsilon1 = epsilon_air, double epsilon2 = epsilon_substrate) {
    double large_sqrt = sqrt(epsilon2 / epsilon1) * sqrt(1 - (epsilon1 / epsilon2) * sin_theta_inc * sin_theta_inc);
    double Gamma_r = (cos_theta_inc - large_sqrt) / (cos_theta_inc + large_sqrt);
    double Gamma_t = (2 * cos_theta_inc) / (cos_theta_inc + large_sqrt);
    return {Gamma_r, Gamma_t};
}


pair<double, double> fresnel_coeffs_TM(double cos_theta_inc, double sin_theta_inc, double epsilon1 = epsilon_air, double epsilon2 = epsilon_substrate) {
    double large_sqrt = sqrt(epsilon1 / epsilon2) * sqrt(1 - (epsilon1 / epsilon2) * sin_theta_inc * sin_theta_inc);
    double Gamma_r = (-cos_theta_inc + large_sqrt) / (cos_theta_inc + large_sqrt);
    double Gamma_t = (2 * sqrt(epsilon1 / epsilon2) * cos_theta_inc) / (cos_theta_inc + large_sqrt);
    return {Gamma_r, Gamma_t};
}


// =========================================
//  Decomposition of E polarization
// =========================================
tuple<double, double, double> polarization_decomposition(const Eigen::Vector3cd& E_rot){
    double E0 = E_rot.norm();

    double cos_beta = E_rot[1].real()/E0;
    double sin_beta = sqrt(1 - cos_beta * cos_beta);

    return {E0, cos_beta, sin_beta};
}


// =========================================
//  Reflection and Transmission Fields
// =========================================

Eigen::Vector3cd reflected_field_TE(double Gamma_r, double cos_theta_inc, double sin_theta_inc, const Eigen::Vector3d& x,
                                    double E0, double k1 = k_air) {
    return Eigen::Vector3cd(0.0, Gamma_r * E0 * exp(-j * k1 * (sin_theta_inc * x[0] - cos_theta_inc * x[2])), 0.0);
}

Eigen::Vector3cd reflected_field_TM(double Gamma_r, double cos_theta_inc, double sin_theta_inc, const Eigen::Vector3d& x, 
                                    double E0, double k1 = k_air) {
    complex<double> wave = E0 * Gamma_r * exp(-j * k1 * (x[0] * sin_theta_inc - x[2] * cos_theta_inc));
    Eigen::Vector3cd E_ref (cos_theta_inc * wave, 0.0, sin_theta_inc * wave);
    return E_ref;
}


Eigen::Vector3cd transmitted_field_TE(double Gamma_t, double sin_theta_inc, const Eigen::Vector3d& x, double E0, 
                                      double k1 = k_air, double k2 = k_substrate) {
    double sin_theta_trans = k1 / k2 * sin_theta_inc;
    double cos_theta_trans = sqrt(1 - sin_theta_trans * sin_theta_trans);
    
    return {Eigen::Vector3cd(0.0, Gamma_t * E0 * exp(-j * k2 * (sin_theta_trans * x[0] + cos_theta_trans * x[2])), 0.0)
    };
}

Eigen::Vector3cd transmitted_field_TM(double Gamma_t, double sin_theta_inc, const Eigen::Vector3d& x, double E0, 
                                      double k1 = k_air, double k2 = k_substrate) {
    double sin_theta_trans = k1 / k2 * sin_theta_inc;
    double cos_theta_trans = sqrt(1 - sin_theta_trans * sin_theta_trans);

    complex<double> wave = E0 * Gamma_t * exp(-j * k2 * (x[0] * sin_theta_trans + x[2] * cos_theta_trans));

    return {Eigen::Vector3cd(wave * cos_theta_trans, 0.0, - wave * sin_theta_trans)
    };
}

// =========================================
//  Driver Function
// =========================================

Eigen::Vector3cd driver(const Eigen::Vector3d& k_inc, const Eigen::Vector3cd& E_inc, const Eigen::Vector3d& x) {
    auto cosP = cos_phi(k_inc);
    auto sinP = sin_phi(k_inc);

    auto Rz = rotation_matrix_z(cosP, sinP);

    Eigen::Vector3d k_rot = Rz * k_inc;
    Eigen::Vector3cd E_rot = Rz * E_inc;

    double cos_theta_inc = cos_theta(k_rot);
    double sin_theta_inc = sin_theta(k_rot);

    auto [E0, cos_beta, sin_beta] = polarization_decomposition(E_rot);

    auto [Gamma_r_TE, Gamma_t_TE] = fresnel_coeffs_TE(cos_theta_inc, sin_theta_inc);
    auto [Gamma_r_TM, Gamma_t_TM] = fresnel_coeffs_TM(cos_theta_inc, sin_theta_inc);

    auto E_ref_TE = reflected_field_TE(Gamma_r_TE, cos_theta_inc, sin_theta_inc, x, E0);
    auto E_ref_TM = reflected_field_TM(Gamma_r_TM, cos_theta_inc, sin_theta_inc, x, E0);
    auto E_trans_TE = transmitted_field_TE(Gamma_t_TE, sin_theta_inc, x, E0);
    auto E_trans_TM = transmitted_field_TM(Gamma_t_TM, sin_theta_inc, x, E0);

    auto Rz_inv = rotation_matrix_z_inv(cosP, sinP);

    Eigen::Vector3cd E_tot_1 = E_inc + Rz_inv * (cos_beta * E_ref_TE + sin_beta * E_ref_TM);
    Eigen::Vector3cd E_tot_2 = Rz_inv * (cos_beta * E_trans_TE + sin_beta * E_trans_TM);

    return (x[2] > 0) ? E_tot_2 : E_tot_1;
}

// =========================================
//  Main Function
// =========================================
int main() {
    Eigen::Vector3d k_inc(1.0, 0.0, 2.0);
    Eigen::Vector3cd E_inc(2.0, -1.0, -1.0);
    Eigen::Vector3d x(2.0, 2.0, 2.0);

    auto E_tot = driver(k_inc, E_inc, x);

    cout << "E_tot: (" << E_tot[0] << ", " << E_tot[1] << ", " << E_tot[2] << ")" << endl;

    return 0;
}
