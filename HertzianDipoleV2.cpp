#include <iostream>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include "fieldDipole.h"
#include "computeAngles.h"
#include "Constants.h"

using namespace std;


struct Constants constants;

const complex<double> j(0, 1);  // Imaginary unit
const double eta0 = constants.eta0;
const double k0 = constants.k0;

// =========================================
//  Compute Angular Components (Pass by Reference)
// =========================================
void computeAngles(const Eigen::Vector3d& x, double r, 
                           double& cosTheta, double& sinTheta, 
                           double& cosPhi, double& sinPhi) {
    if (r == 0) {
        cosTheta = sinTheta = 0.0;
        cosPhi = 1.0; sinPhi = 0.0;
        return;
    }

    cosTheta = x[2] / r;
    sinTheta = sqrt(1 - cosTheta * cosTheta);
    double xy_norm = std::hypot(x[0], x[1]);

    if (xy_norm == 0) {
        cosPhi = 1.0;
        sinPhi = 0.0;
    } else {
        cosPhi = x[0] / xy_norm;
        sinPhi = x[1] / xy_norm;
    }
}

// =========================================
//  Compute E-Field Components (Spherical, Precompute E_r & E_theta)
// =========================================
void compute_E_fields(double r, double Iel, double cos_theta, double sin_theta, 
                      const complex<double>& expK0r, complex<double>& E_r_out, 
                      complex<double>& E_theta_out) {
    if (r == 0) {
        E_r_out = E_theta_out = 0.0;
        return;
    }

    E_r_out = eta0 * Iel * cos_theta / (2.0 * constants.pi * r * r) 
              * (1.0 + 1.0 / (j * k0 * r)) * expK0r;
    
    E_theta_out = (j * eta0 * Iel * sin_theta / (4.0 * constants.pi * r)) 
                  * (1.0 + 1.0 / (j * k0 * r) - 1.0 / (k0 * r * r)) * expK0r;
}

// =========================================
//  Compute E-Field Components (Cartesian, No Redundant Calls)
// =========================================
Eigen::Vector3cd compute_E_cartesian(double sin_theta, double cos_theta, 
                                     double sin_phi, double cos_phi, 
                                     const complex<double>& E_r, 
                                     const complex<double>& E_theta) {
    return {
        E_r * sin_theta * cos_phi + E_theta * cos_theta * cos_phi,
        E_r * sin_theta * sin_phi + E_theta * cos_theta * sin_phi,
        E_r * cos_theta - E_theta * sin_theta
    };
}

// =========================================
//  Compute H-Field Components (Precompute H_phi)
// =========================================
void compute_H_fields(double r, double Iel, double sin_theta, 
                      const complex<double>& expK0r, complex<double>& H_phi_out) {
    if (r == 0) {
        H_phi_out = 0.0;
        return;
    }

    H_phi_out = j * k0 * Iel * sin_theta / (4.0 * constants.pi * r) 
                * (1.0 + 1.0 / (j * k0 * r)) * expK0r;
}

// =========================================
//  Compute Both E and H Fields (Precompute Values)
// =========================================
pair<Eigen::Vector3cd, Eigen::Vector3cd> fieldDipole(double r, double Iel, 
    double cos_theta, double sin_theta, double cos_phi, double sin_phi, 
    const complex<double>& expK0r) {

    // Precompute E_r and E_theta
    complex<double> E_r_val, E_theta_val;
    compute_E_fields(r, Iel, cos_theta, sin_theta, expK0r, E_r_val, E_theta_val);

    // Compute E-field in Cartesian coordinates
    Eigen::Vector3cd E = compute_E_cartesian(sin_theta, cos_theta, sin_phi, cos_phi, E_r_val, E_theta_val);

    // Precompute H_phi
    complex<double> H_phi_val;
    compute_H_fields(r, Iel, sin_theta, expK0r, H_phi_val);

    // Compute H-field
    Eigen::Vector3cd H = {
        -H_phi_val * sin_phi,
        H_phi_val * cos_phi,
        0.0
    };

    return {E, H};
}

// =========================================
//  Main Function
// =========================================
int main() {
    double Iel = 5.0;
    Eigen::Vector3d x = {0.1, 0.1, 0.1}; // Test position
    double r = x.norm();
    double cos_theta = 0.0, sin_theta = 0.0, cos_phi = 0.0, sin_phi = 0.0;
    
    computeAngles(x, r, cos_theta, sin_theta, cos_phi, sin_phi);
    
    // Compute exp(-j * k0 * r) once
    complex<double> expK0r = std::polar(1.0, -k0 * r);

    // Compute E and H fields
    auto [E, H] = fieldDipole(r, Iel, cos_theta, sin_theta, cos_phi, sin_phi, expK0r);

    // Print results
    cout << "E-field: (" << E[0] << ", " << E[1] << ", " << E[2] << ")" << endl;
    cout << "H-field: (" << H[0] << ", " << H[1] << ", " << H[2] << ")" << endl;

    return 0;
}
