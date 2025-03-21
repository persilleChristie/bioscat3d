#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <cmath>

using namespace std;
using namespace Eigen;

const complex<double> j(0, 1); // Imaginary unit

// =========================================================
//  Original Python Code (Outcommented)
// =========================================================
/*
import numpy as np

j = complex(0,1) # imaginary unit

# Constants
omega = 1
epsilon0 = 1
n0 = 1
mu0 = 1
k0 = omega * np.sqrt(epsilon0 * mu0)

epsilon_air = epsilon0
epsilon_substrate = 2
n_air = n0
n_substrate = 2
k_air = n_air * k0
k_substrate = n_substrate * k0
*/

const double omega = 1;
const double epsilon0 = 1;
const double n0 = 1;
const double mu0 = 1;
const double k0 = omega * sqrt(epsilon0 * mu0);

const double epsilon_air = epsilon0;
const double epsilon_substrate = 2;
const double n_air = n0;
const double n_substrate = 2;
const double k_air = n_air * k0;
const double k_substrate = n_substrate * k0;

// Unit vectors
const Vector3d xhat(1, 0, 0);
const Vector3d yhat(0, 1, 0);
const Vector3d zhat(0, 0, 1);

// =========================================================
//  Compute Azimuthal Angle
// =========================================================
/*
def azimutal_angle(k_inc):
    cos_phi = k_inc[0]/np.sqrt(k_inc[0]**2 + k_inc[1]**2)
    sin_phi = np.sqrt(1 - cos_phi**2)
    return cos_phi, sin_phi
*/
pair<double, double> azimutal_angle(const Vector3d& k_inc) {
    double cos_phi = k_inc[0] / hypot(k_inc[0], k_inc[1]);
    double sin_phi = sqrt(1 - cos_phi * cos_phi);
    return {cos_phi, sin_phi};
}

// =========================================================
//  Rotation Matrix
// =========================================================
/*
def rotate(k_inc, E_inc, cos_phi, sin_phi):
    Rz = np.array([[cos_phi, -sin_phi, 0],
                    [sin_phi, cos_phi, 0],
                    [0, 0, 1]])
    k_rot = Rz @ k_inc
    E_rot = lambda r : Rz @ E_inc(r)
    return k_rot, E_rot 
*/
pair<Vector3d, function<Vector3d(Vector3d)>> rotate(const Vector3d& k_inc, function<Vector3d(Vector3d)> E_inc, double cos_phi, double sin_phi) {
    Matrix3d Rz;
    Rz << cos_phi, -sin_phi, 0,
          sin_phi, cos_phi, 0,
          0, 0, 1;

    Vector3d k_rot = Rz * k_inc;
    auto E_rot = [Rz, E_inc](const Vector3d& r) {
        return Rz * E_inc(r);
    };
    return {k_rot, E_rot};
}

// =========================================================
//  Compute Polar Angle
// =========================================================
/*
def polar_angle(k_rot):
    cos_theta = k_rot[2] / np.sqrt(k_rot[0]**2 + k_rot[2]**2)
    sin_theta = np.sqrt(1 - cos_theta**2)
    return cos_theta, sin_theta
*/
pair<double, double> polar_angle(const Vector3d& k_rot) {
    double cos_theta = k_rot[2] / hypot(k_rot[0], k_rot[2]);
    double sin_theta = sqrt(1 - cos_theta * cos_theta);
    return {cos_theta, sin_theta};
}

// =========================================================
//  Compute Fresnel Coefficients for TE & TM Modes
// =========================================================
/*
def fresnel_coeffs_TE(cos_theta_inc, sin_theta_inc, epsilon1 = epsilon_air, epsilon2 = epsilon_substrate):
    large_sqrt = np.sqrt(epsilon2/epsilon1) * np.sqrt(1 - (epsilon1/epsilon2)*sin_theta_inc**2)
    Gamma_r = (cos_theta_inc - large_sqrt) / (cos_theta_inc + large_sqrt)
    Gamma_t = (2*cos_theta_inc) / (cos_theta_inc + large_sqrt)
    return Gamma_r, Gamma_t
*/
pair<complex<double>, complex<double>> fresnel_coeffs_TE(double cos_theta_inc, double sin_theta_inc, double epsilon1 = epsilon_air, double epsilon2 = epsilon_substrate) {
    complex<double> large_sqrt = sqrt(epsilon2 / epsilon1) * sqrt(1.0 - (epsilon1 / epsilon2) * pow(sin_theta_inc, 2));
    complex<double> Gamma_r = (cos_theta_inc - large_sqrt) / (cos_theta_inc + large_sqrt);
    complex<double> Gamma_t = (2.0 * cos_theta_inc) / (cos_theta_inc + large_sqrt);
    return {Gamma_r, Gamma_t};
}

/*
def fresnel_coeffs_TM(cos_theta_inc, sin_theta_inc, epsilon1 = epsilon_air, epsilon2 = epsilon_substrate):
    large_sqrt = np.sqrt(epsilon1/epsilon2) * np.sqrt(1 - (epsilon1/epsilon2)*sin_theta_inc**2)
    Gamma_r = (-cos_theta_inc + large_sqrt) / (cos_theta_inc + large_sqrt)
    Gamma_t = (2 * np.sqrt(epsilon1/epsilon2) * cos_theta_inc) / (cos_theta_inc + large_sqrt)
    return Gamma_r, Gamma_t
*/
pair<complex<double>, complex<double>> fresnel_coeffs_TM(double cos_theta_inc, double sin_theta_inc, double epsilon1 = epsilon_air, double epsilon2 = epsilon_substrate) {
    complex<double> large_sqrt = sqrt(epsilon1 / epsilon2) * sqrt(1.0 - (epsilon1 / epsilon2) * pow(sin_theta_inc, 2));
    complex<double> Gamma_r = (-cos_theta_inc + large_sqrt) / (cos_theta_inc + large_sqrt);
    complex<double> Gamma_t = (2.0 * sqrt(epsilon1 / epsilon2) * cos_theta_inc) / (cos_theta_inc + large_sqrt);
    return {Gamma_r, Gamma_t};
}

// =========================================================
//  Original Python Code (Outcommented)
// =========================================================
/*
def reflected_field_TE(Gamma_r, cos_theta_inc, sin_theta_inc, E0, k1 = k_air):
    E_ref = lambda r : (yhat * Gamma_r * E0(r) 
                        * np.exp(- j * k1 * (r[0] * sin_theta_inc - r[2] * cos_theta_inc)))
    return E_ref
*/
// C++ Version:
auto reflected_field_TE(complex<double> Gamma_r, double cos_theta_inc, double sin_theta_inc, 
    function<double(Vector3d)> E0, double k1 = k_air) {
return [Gamma_r, cos_theta_inc, sin_theta_inc, E0, k1](const Vector3d& r) {
return yhat * real(Gamma_r * E0(r) * exp(-j * k1 * (r[0] * sin_theta_inc - r[2] * cos_theta_inc)));
};
}

/*
def reflected_field_TM(Gamma_r, cos_theta_inc, sin_theta_inc, E0, k1 = k_air):
E_ref = lambda r : ((xhat * cos_theta_inc + zhat * sin_theta_inc) * Gamma_r * E0(r) 
    * np.exp(- j * k1 * (r[0] * sin_theta_inc - r[2] * cos_theta_inc)))
return E_ref
*/
// C++ Version:
auto reflected_field_TM(complex<double> Gamma_r, double cos_theta_inc, double sin_theta_inc, 
    function<double(Vector3d)> E0, double k1 = k_air) {
return [Gamma_r, cos_theta_inc, sin_theta_inc, E0, k1](const Vector3d& r) {
return (xhat * cos_theta_inc + zhat * sin_theta_inc) * 
real(Gamma_r * E0(r) * exp(-j * k1 * (r[0] * sin_theta_inc - r[2] * cos_theta_inc)));
};
}

/*
def transmitted_field_TE(Gamma_t, sin_theta_inc, E0, k1 = k_air, k2 = k_substrate):
sin_theta_trans = k1/k2 * sin_theta_inc
cos_theta_trans = np.sqrt(1 - sin_theta_trans**2)

E_trans = lambda r : (yhat * Gamma_t * E0(r) 
    * np.exp(- j * k2 * (r[0] * sin_theta_trans + r[2] * cos_theta_trans)))
return E_trans
*/
// C++ Version:
auto transmitted_field_TE(complex<double> Gamma_t, double sin_theta_inc, function<double(Vector3d)> E0, 
      double k1 = k_air, double k2 = k_substrate) {
double sin_theta_trans = (k1 / k2) * sin_theta_inc;
double cos_theta_trans = sqrt(1 - sin_theta_trans * sin_theta_trans);

return [Gamma_t, sin_theta_trans, cos_theta_trans, E0, k2](const Vector3d& r) {
return yhat * real(Gamma_t * E0(r) * exp(-j * k2 * (r[0] * sin_theta_trans + r[2] * cos_theta_trans)));
};
}

/*
def transmitted_field_TM(Gamma_t, sin_theta_inc, E0, k1 = k_air, k2 = k_substrate):
sin_theta_trans = k1/k2 * sin_theta_inc
cos_theta_trans = np.sqrt(1 - sin_theta_trans**2)

E_trans = lambda r : ((xhat * cos_theta_trans - zhat * sin_theta_trans) * Gamma_t * E0(r)
    * np.exp(- j * k2 * (r[0] * sin_theta_trans + r[2] * cos_theta_trans)))
return E_trans
*/
// C++ Version:
auto transmitted_field_TM(complex<double> Gamma_t, double sin_theta_inc, function<double(Vector3d)> E0, 
      double k1 = k_air, double k2 = k_substrate) {
double sin_theta_trans = (k1 / k2) * sin_theta_inc;
double cos_theta_trans = sqrt(1 - sin_theta_trans * sin_theta_trans);

return [Gamma_t, sin_theta_trans, cos_theta_trans, E0, k2](const Vector3d& r) {
return (xhat * cos_theta_trans - zhat * sin_theta_trans) * 
real(Gamma_t * E0(r) * exp(-j * k2 * (r[0] * sin_theta_trans + r[2] * cos_theta_trans)));
};
}



// =========================================================
//  Main Function
// =========================================================
int main() {
    Vector3d k_inc(1, 0, 2);
    auto E_inc = [](const Vector3d& r) {
        return Vector3d(2, -1, -1);
    };

    auto [cos_phi, sin_phi] = azimutal_angle(k_inc);
    auto [k_rot, E_rot] = rotate(k_inc, E_inc, cos_phi, sin_phi);
    auto [cos_theta_inc, sin_theta_inc] = polar_angle(k_rot);

    auto [Gamma_r_TE, Gamma_t_TE] = fresnel_coeffs_TE(cos_theta_inc, sin_theta_inc);
    auto [Gamma_r_TM, Gamma_t_TM] = fresnel_coeffs_TM(cos_theta_inc, sin_theta_inc);

    cout << "TE Reflection Coefficient: " << Gamma_r_TE << endl;
    cout << "TE Transmission Coefficient: " << Gamma_t_TE << endl;
    cout << "TM Reflection Coefficient: " << Gamma_r_TM << endl;
    cout << "TM Transmission Coefficient: " << Gamma_t_TM << endl;

    Vector3d r(2, 2, 2);  // Test position
    complex<double> Gamma_r_TE(0.5, 0.1);
    complex<double> Gamma_t_TE(0.8, 0.05);
    complex<double> Gamma_r_TM(0.4, 0.2);
    complex<double> Gamma_t_TM(0.7, 0.03);
    
    auto E0 = [](const Vector3d& r) { return 1.0; };  // E0 function

    auto E_ref_TE = reflected_field_TE(Gamma_r_TE, 0.5, 0.6, E0);
    auto E_ref_TM = reflected_field_TM(Gamma_r_TM, 0.5, 0.6, E0);
    auto E_trans_TE = transmitted_field_TE(Gamma_t_TE, 0.5, E0);
    auto E_trans_TM = transmitted_field_TM(Gamma_t_TM, 0.5, E0);

    Vector3d E_TE = E_ref_TE(r);
    Vector3d E_TM = E_ref_TM(r);
    Vector3d E_TTE = E_trans_TE(r);
    Vector3d E_TTM = E_trans_TM(r);

    cout << "Reflected TE E-field: " << E_TE.transpose() << endl;
    cout << "Reflected TM E-field: " << E_TM.transpose() << endl;
    cout << "Transmitted TE E-field: " << E_TTE.transpose() << endl;
    cout << "Transmitted TM E-field: " << E_TTM.transpose() << endl;

    return 0;
}
