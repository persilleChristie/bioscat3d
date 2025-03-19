#include <iostream>
#include <cmath>
#include <complex>
#include <array>
#include <vector>

using namespace std;

const complex<double> j(0, 1);  // Imaginary unit

// =========================================
//  Constants Definition
// =========================================
/*
omega = 1
epsilon0 = 1
n0 = 1
mu0 = 1
k0 = omega * np.sqrt(epsilon0 * mu0)
*/

const double omega = 1.0;
const double epsilon0 = 1.0;
const double n0 = 1.0;
const double mu0 = 1.0;
const double k0 = omega * sqrt(epsilon0 * mu0);

/*
epsilon_air = epsilon0
epsilon_substrate = 2
n_air = n0
n_substrate = 2
k_air = n_air * k0
k_substrate = n_substrate * k0
*/

const double epsilon_air = epsilon0;
const double epsilon_substrate = 2.0;
const double n_air = n0;
const double n_substrate = 2.0;
const double k_air = n_air * k0;
const double k_substrate = n_substrate * k0;

// =========================================
//  Cartesian Unit Vectors
// =========================================
/*
xhat = np.array([1,0,0])
yhat = np.array([0,1,0])
zhat = np.array([0,0,1])
*/

const array<double, 3> xhat = {1.0, 0.0, 0.0};
const array<double, 3> yhat = {0.0, 1.0, 0.0};
const array<double, 3> zhat = {0.0, 0.0, 1.0};

// =========================================
//  Azimuthal Angle Computation
// =========================================
/*
def azimutal_angle(k_inc):
    cos_phi = k_inc[0]/np.sqrt(k_inc[0]**2 + k_inc[1]**2)
    sin_phi = np.sqrt(1 - cos_phi**2)
*/

inline double cos_phi(const array<double, 3>& k_inc) {
    double xy_norm = hypot(k_inc[0], k_inc[1]);
    return (xy_norm == 0) ? 1.0 : k_inc[0] / xy_norm;
}

inline double sin_phi(const array<double, 3>& k_inc) {
    double xy_norm = hypot(k_inc[0], k_inc[1]);
    return (xy_norm == 0) ? 0.0 : k_inc[1] / xy_norm;
}

// =========================================
//  Rotation Matrix (Z-axis)
// =========================================
/*
def rotate(k_inc, E_inc, cos_phi, sin_phi):
    Rz = np.array([[cos_phi, -sin_phi, 0],
                    [sin_phi, cos_phi, 0],
                    [0, 0, 1]])
*/

array<array<double, 3>, 3> rotation_matrix_z(double cos_phi, double sin_phi) {
    return {{
        {cos_phi, -sin_phi, 0},
        {sin_phi, cos_phi,  0},
        {0,       0,        1}
    }};
}

// =========================================
//  Polar Angle Computation
// =========================================
/*
def polar_angle(k_rot):
    cos_theta = k_rot[2] / np.sqrt(k_rot[0]**2 + k_rot[2]**2)
    sin_theta = np.sqrt(1 - cos_theta**2)
*/

inline double cos_theta(const array<double, 3>& k_rot) {
    return k_rot[2] / hypot(k_rot[0], k_rot[2]);
}

inline double sin_theta(const array<double, 3>& k_rot) {
    double cosT = cos_theta(k_rot);
    return sqrt(1 - cosT * cosT);
}

// =========================================
//  Fresnel Coefficients (TE and TM)
// =========================================
/*
def fresnel_coeffs_TE(cos_theta_inc, sin_theta_inc, epsilon1 = epsilon_air, epsilon2 = epsilon_substrate):
*/

pair<double, double> fresnel_coeffs_TE(double cos_theta_inc, double sin_theta_inc, double epsilon1 = epsilon_air, double epsilon2 = epsilon_substrate) {
    double large_sqrt = sqrt(epsilon2 / epsilon1) * sqrt(1 - (epsilon1 / epsilon2) * sin_theta_inc * sin_theta_inc);
    double Gamma_r = (cos_theta_inc - large_sqrt) / (cos_theta_inc + large_sqrt);
    double Gamma_t = (2 * cos_theta_inc) / (cos_theta_inc + large_sqrt);
    return {Gamma_r, Gamma_t};
}

/*
def fresnel_coeffs_TM(cos_theta_inc, sin_theta_inc, epsilon1 = epsilon_air, epsilon2 = epsilon_substrate):
*/

pair<double, double> fresnel_coeffs_TM(double cos_theta_inc, double sin_theta_inc, double epsilon1 = epsilon_air, double epsilon2 = epsilon_substrate) {
    double large_sqrt = sqrt(epsilon1 / epsilon2) * sqrt(1 - (epsilon1 / epsilon2) * sin_theta_inc * sin_theta_inc);
    double Gamma_r = (-cos_theta_inc + large_sqrt) / (cos_theta_inc + large_sqrt);
    double Gamma_t = (2 * sqrt(epsilon1 / epsilon2) * cos_theta_inc) / (cos_theta_inc + large_sqrt);
    return {Gamma_r, Gamma_t};
}

// =========================================
//  Reflection and Transmission Fields
// =========================================
/*
def reflected_field_TE(Gamma_r, cos_theta_inc, sin_theta_inc, E0, k1 = k_air):
*/

array<double, 3> reflected_field_TE(double Gamma_r, double cos_theta_inc, double sin_theta_inc, double E0, double k1 = k_air) {
    return {{
        0.0,
        Gamma_r * E0 * exp(-j * k1 * (sin_theta_inc - cos_theta_inc)),
        0.0
    }};
}

/*
def transmitted_field_TE(Gamma_t, sin_theta_inc, E0, k1 = k_air, k2 = k_substrate):
*/

array<double, 3> transmitted_field_TE(double Gamma_t, double sin_theta_inc, double E0, double k1 = k_air, double k2 = k_substrate) {
    double sin_theta_trans = k1 / k2 * sin_theta_inc;
    double cos_theta_trans = sqrt(1 - sin_theta_trans * sin_theta_trans);
    return {{
        0.0,
        Gamma_t * E0 * exp(-j * k2 * (sin_theta_trans + cos_theta_trans)),
        0.0
    }};
}

// =========================================
//  Driver Function
// =========================================
/*
def driver(k_inc, E_inc):
*/

array<double, 3> driver(const array<double, 3>& k_inc, double E_inc) {
    double cosP = cos_phi(k_inc);
    double sinP = sin_phi(k_inc);

    auto Rz = rotation_matrix_z(cosP, sinP);

    double cos_theta_inc = cos_theta(k_inc);
    double sin_theta_inc = sin_theta(k_inc);

    auto [Gamma_r_TE, Gamma_t_TE] = fresnel_coeffs_TE(cos_theta_inc, sin_theta_inc);
    auto [Gamma_r_TM, Gamma_t_TM] = fresnel_coeffs_TM(cos_theta_inc, sin_theta_inc);

    auto E_ref_TE = reflected_field_TE(Gamma_r_TE, cos_theta_inc, sin_theta_inc, E_inc);
    auto E_trans_TE = transmitted_field_TE(Gamma_t_TE, sin_theta_inc, E_inc);

    return E_ref_TE;
}

// =========================================
//  Main Function
// =========================================
int main() {
    array<double, 3> k_inc = {1.0, 0.0, 2.0};
    double E_inc = 2.0;

    auto E_tot = driver(k_inc, E_inc);

    cout << "E_tot: (" << E_tot[0] << ", " << E_tot[1] << ", " << E_tot[2] << ")" << endl;

    return 0;
}
