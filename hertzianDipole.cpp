#include <iostream>
#include <cmath>
#include <complex>
#include <algorithm>
#include <Eigen/Dense>


//g++ hertzianDipole.cpp -o hD.exe -I"D:/vcpkg/installed/x64-windows/include/eigen3" //

using namespace std;

const complex<double> j(0, 1);  // Imaginary unit
const double eta0 = 1.0;
const double k0 = 1.0;

// =========================================
//  Compute Angular Components
// =========================================

inline void compute_angles(const Eigen::Vector3d& x, double r, double& cosTheta, double& sinTheta, double& cosPhi, double& sinPhi) {
    if (r == 0) {
        cosTheta = 0.0;
        sinTheta = 0.0;
        cosPhi = 1.0; // Default to x-axis
        sinPhi = 0.0;
        return;
    }

    cosTheta = x[2] / r;
    sinTheta = sqrt(1 - cosTheta * cosTheta); // Avoid separate call to cosTheta
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
//  Compute E-Field Components (Spherical)
// =========================================
complex<double> E_r(const Eigen::Vector3d& x, double r, double Iel, double cos_theta) {
    return (r == 0) ? 0.0 : eta0 * Iel * cos_theta / (2.0 * M_PI * r * r) * (1.0 + 1.0 / (j * k0 * r)) * exp(-j * k0 * r);
}

complex<double> E_theta(const Eigen::Vector3d& x, double r, double Iel, double sin_theta) {
    return (r == 0) ? 0.0 : (j * eta0 * Iel * sin_theta / (4.0 * M_PI * r)) *
           (1.0 + 1.0 / (j * k0 * r) - 1.0 / (k0 * r * r)) * exp(-j * k0 * r);
}

// =========================================
//  Compute E-Field Components Cartesian cooerdinates
// =========================================
complex<double> E_x(const Eigen::Vector3d& x, double r, double Iel, double cos_theta, double sin_theta, double cos_phi, double sin_phi) {
    return E_r(x, r, Iel, cos_theta) * sin_theta * cos_phi + E_theta(x, r, Iel, sin_theta) * cos_theta * cos_phi;
}

complex<double> E_y(const Eigen::Vector3d& x, double r, double Iel, double cos_theta, double sin_theta, double cos_phi, double sin_phi) {
    return E_r(x, r, Iel, cos_theta) * sin_theta * sin_phi + E_theta(x, r, Iel, sin_theta) * cos_theta * sin_phi;
}

complex<double> E_z(const Eigen::Vector3d& x, double r, double Iel, double cos_theta, double sin_theta) {
    return E_r(x, r, Iel, cos_theta) * cos_theta - E_theta(x, r, Iel, sin_theta) * sin_theta;
}

// =========================================
//  Compute H-Field Components (Spherical)
// =========================================
complex<double> H_phi(const Eigen::Vector3d& x, double r, double Iel, double sin_theta) {
    return (r == 0) ? 0.0 : j * k0 * Iel * sin_theta / (4.0 * M_PI * r) * (1.0 + 1.0 / (j * k0 * r)) * exp(-j * k0 * r);
}

// =========================================
//  Compute H-Field Components Cartesian cooerdinates
// =========================================
complex<double> H_x(const Eigen::Vector3d& x, double r, double Iel, double cos_theta, double sin_theta, double cos_phi, double sin_phi) {
    return - H_phi(x, r, Iel, sin_theta) * sin_phi;
}

complex<double> H_y(const Eigen::Vector3d& x, double r, double Iel, double cos_theta, double sin_theta, double cos_phi, double sin_phi) {
    return H_phi(x, r, Iel, sin_theta) * cos_phi;
}

// =========================================
//  Compute Both E and H Fields (Spherical)
// =========================================
pair<Eigen::Vector3cd, Eigen::Vector3cd> fieldDipole(const Eigen::Vector3d& x, double r, double Iel, double cos_theta, double sin_theta, double cos_phi, double sin_phi) {
    compute_angles(x, r, cos_theta, sin_theta, cos_phi, sin_phi);
    Eigen::Vector3cd E = {E_x(x, r, Iel, cos_theta, sin_theta, cos_phi, sin_phi), E_y(x, r, Iel, cos_theta, sin_theta, cos_phi, sin_phi), E_z(x, r, Iel, cos_theta, sin_theta)};    
    Eigen::Vector3cd H = {H_x(x, r, Iel, cos_theta, sin_theta, cos_phi, sin_phi), H_y(x, r, Iel, cos_theta, sin_theta, cos_phi, sin_phi), 0.0};
    return {E, H};
}

// =========================================
//  Main Function for Testing
// =========================================
int main() {
    double Iel = 5.0;
    Eigen::Vector3d x = {0.1, 0.1, 0.1}; // Test position
    double r = x.norm();
    double cos_theta = 0.0;
    double sin_theta = 0.0;
    double cos_phi = 0.0;
    double sin_phi = 0.0;

    
    // Compute E and H fields
    auto [E, H] = fieldDipole(x, r, Iel, cos_theta, sin_theta, cos_phi, sin_phi);
    

    // Print results
    cout << "E-field: (" << E[0] << ", " << E[1] << ", " << E[2] << ")" << endl;
    cout << "H-field: (" << H[0] << ", " << H[1] << ", " << H[2] << ")" << endl;

    //printf("sin_theta: %f\n", sin_theta(x));
    //printf("cos_phi: %f\n", cos_phi(x));
    //printf("sin_phi: %f\n", sin_phi(x));
    //printf("H_phi: %f\n", H_phi(x, Iel).real());
    //printf("H_x: %f\n", H_x(x, Iel).real());
    //printf("H_y: %f\n", H_y(x, Iel).real());


    
    
    return 0;
}
