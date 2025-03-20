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
//  Compute Fresnel Coefficients
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

// =========================================================
//  Compute Reflected Field TE
// =========================================================
/*
def reflected_field_TE(Gamma_r, cos_theta_inc, sin_theta_inc, E0, k1 = k_air):
    E_ref = lambda r : (yhat * Gamma_r * E0(r) 
                        * np.exp(- j * k1 * (r[0] * sin_theta_inc - r[2] * cos_theta_inc)))
    return E_ref
*/
auto reflected_field_TE(complex<double> Gamma_r, double cos_theta_inc, double sin_theta_inc, function<double(Vector3d)> E0, double k1 = k_air) {
    return [Gamma_r, cos_theta_inc, sin_theta_inc, E0, k1](const Vector3d& r) {
        return yhat * real(Gamma_r * E0(r) * exp(-j * k1 * (r[0] * sin_theta_inc - r[2] * cos_theta_inc)));
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
    
    auto E_ref_TE = reflected_field_TE(Gamma_r_TE, cos_theta_inc, sin_theta_inc, [](const Vector3d& r) { return 1.0; });

    Vector3d r(2, 2, 2);
    Vector3d E_field = E_ref_TE(r);

    cout << "Reflected E-field: " << E_field.transpose() << endl;
    
    return 0;
}
