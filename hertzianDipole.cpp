#include <iostream>
#include <cmath>
#include <complex>
#include <array>
#include <algorithm>

using namespace std;

const complex<double> j(0, 1);  // Imaginary unit
const double eta0 = 1.0;
const double k0 = 1.0;

// =========================================
//  Compute the norm of a 3D vector
// =========================================
double norm(const array<double, 3>& x) {
    return std::hypot(x[0], x[1], x[2]);
}

// =========================================
//  Compute Angular Components
// =========================================
inline double cos_theta(const array<double, 3>& x) {
    double r = norm(x);
    return (r == 0) ? 0.0 : x[2] / r;
}

inline double sin_theta(const array<double, 3>& x) {
    double cosT = cos_theta(x);
    cosT = std::clamp(cosT, -1.0, 1.0);
    return sqrt(1 - cosT * cosT);
    
}


// Compute cos(φ) and sin(φ)
inline double cos_phi(const array<double, 3>& x) {
    double phi = atan2(x[1], x[0]);
    double cos_phi = cos(phi);
    return cos_phi;
}

// Compute cos(φ) and sin(φ)
inline double sin_phi(const array<double, 3>& x) {
    double phi = atan2(x[1], x[0]);
    double sin_phi = sin(phi);
    return sin_phi;
    
}


// =========================================
//  Compute E-Field Components (Spherical)
// =========================================
complex<double> E_r(const array<double, 3>& x, double Iel) {
    double r = norm(x);
    return (r == 0) ? 0.0 : eta0 * Iel * cos_theta(x) / (2.0 * M_PI * r * r) * (1.0 + 1.0 / (j * k0 * r)) * exp(-j * k0 * r);
}

complex<double> E_theta(const array<double, 3>& x, double Iel) {
    double r = norm(x);
    return (r == 0) ? 0.0 : (j * eta0 * Iel * sin_theta(x) / (4.0 * M_PI * r)) *
           (1.0 + 1.0 / (j * k0 * r) - 1.0 / (k0 * r * r)) * exp(-j * k0 * r);
}

// =========================================
//  Compute E-Field Components Cartesian cooerdinates
// =========================================
complex<double> E_x(const array<double, 3>& x, double Iel) {
    return E_r(x, Iel) * sin_theta(x) * cos_phi(x) + E_theta(x, Iel) * cos_theta(x) * cos_phi(x);
}

complex<double> E_y(const array<double, 3>& x, double Iel) {
    return E_r(x, Iel) * sin_theta(x) * sin_phi(x) + E_theta(x, Iel) * cos_theta(x) * sin_phi(x);
}

complex<double> E_z(const array<double, 3>& x, double Iel) {
    return E_r(x, Iel) * cos_theta(x) - E_theta(x, Iel) * sin_theta(x);
}

// =========================================
//  Compute H-Field Components (Spherical)
// =========================================
complex<double> H_phi(const array<double, 3>& x, double Iel) {
    double r = norm(x);
    return (r == 0) ? 0.0 : j * k0 * Iel * sin_theta(x) / (4.0 * M_PI * r) * (1.0 + 1.0 / (j * k0 * r)) * exp(-j * k0 * r);
}

// =========================================
//  Compute H-Field Components Cartesian cooerdinates
// =========================================
complex<double> H_x(const array<double, 3>& x, double Iel) {
    return - H_phi(x, Iel) * sin_phi(x);
}

complex<double> H_y(const array<double, 3>& x, double Iel) {
    return H_phi(x, Iel) * cos_phi(x);
}

// =========================================
//  Compute Both E and H Fields (Spherical)
// =========================================
pair<array<complex<double>, 3>, array<complex<double>, 3>> fieldDipole(const array<double, 3>& x, double Iel) {
    array<complex<double>, 3> E = {E_x(x, Iel), E_y(x, Iel), E_z(x, Iel)};
    array<complex<double>, 3> H = {H_x(x, Iel), H_y(x, Iel), 0.0};
    return {E, H};
}

// =========================================
//  Main Function for Testing
// =========================================
int main() {
    double Iel = 5.0;
    array<double, 3> x = {0.1, 0.1, 0.1}; // Test position

    // Compute E and H fields
    auto [E, H] = fieldDipole(x, Iel);

    // Print results
    cout << "E-field: (" << E[0] << ", " << E[1] << ", " << E[2] << ")" << endl;
    cout << "H-field: (" << H[0] << ", " << H[1] << ", " << H[2] << ")" << endl;

    //rintf("sin_theta: %f\n", sin_theta(x));
    //printf("cos_phi: %f\n", cos_phi(x));
    //printf("sin_phi: %f\n", sin_phi(x));
    //printf("H_phi: %f\n", H_phi(x, Iel).real());
    //printf("H_x: %f\n", H_x(x, Iel).real());
    //printf("H_y: %f\n", H_y(x, Iel).real());


    
    
    return 0;
}
