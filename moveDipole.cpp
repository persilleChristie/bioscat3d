#include <iostream>
#include <cmath>
#include <array>
#include <vector>

using namespace std;

// =========================================
//  Function to Move and Rotate Dipole Fields
// =========================================
/*
import numpy as np

def moveDipoleField(xPrime, rPrime , E, H ):
    # rotate the field with angle between zaxis and r
    cos_theta = rPrime[2] / np.linalg.norm(rPrime)
    sin_theta = np.sqrt(1 - cos_theta**2)
*/

inline double norm(const array<double, 3>& x) {
    return std::hypot(x[0], x[1], x[2]);
}

inline double cos_theta(const array<double, 3>& rPrime) {
    return rPrime[2] / norm(rPrime);
}

inline double sin_theta(const array<double, 3>& rPrime) {
    double cosT = cos_theta(rPrime);
    return sqrt(1 - cosT * cosT);
}

/*
    cos_phi = np.sign(rPrime[1]) * rPrime[0] / np.sqrt(rPrime[0]**2 + rPrime[1]**2)
    sin_phi = np.sqrt(1 - cos_phi**2)
*/

inline double cos_phi(const array<double, 3>& rPrime) {
    double xy_norm = std::hypot(rPrime[0], rPrime[1]);
    return (xy_norm == 0) ? 1.0 : rPrime[0] / xy_norm;
}

inline double sin_phi(const array<double, 3>& rPrime) {
    double xy_norm = std::hypot(rPrime[0], rPrime[1]);
    return (xy_norm == 0) ? 0.0 : rPrime[1] / xy_norm;
}

// =========================================
//  Compute Rotation Matrices
// =========================================
/*
    # rotation matrix for theta around y axis
    Ry = np.array([[cos_theta, 0, sin_theta],
                   [0, 1, 0],
                   [-sin_theta, 0, cos_theta]])
*/

array<array<double, 3>, 3> rotation_matrix_y(const array<double, 3>& rPrime) {
    double cosT = cos_theta(rPrime);
    double sinT = sin_theta(rPrime);
    return {{
        { cosT,  0, sinT },
        { 0,     1,   0  },
        { -sinT, 0, cosT }
    }};
}

/*
    # rotation matrix for phi around z axis
    Rz = np.array([[cos_phi, -sin_phi, 0],
                   [sin_phi, cos_phi, 0],
                   [0, 0, 1]])
*/

array<array<double, 3>, 3> rotation_matrix_z(const array<double, 3>& rPrime) {
    double cosP = cos_phi(rPrime);
    double sinP = sin_phi(rPrime);
    return {{
        { cosP, -sinP, 0 },
        { sinP,  cosP, 0 },
        { 0,      0,   1 }
    }};
}

// =========================================
//  Rotate the Field Components
// =========================================
/*
    # rotate the field
    E_rot = lambda x: np.dot(Rz, np.dot(Ry, E(x)))
    H_rot = lambda x: np.dot(Rz, np.dot(Ry, H(x)))
*/

array<double, 3> rotate_vector(const array<double, 3>& v, const array<array<double, 3>, 3>& R) {
    return {{
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]
    }};
}

// =========================================
//  Compute Final Translated Fields
// =========================================
/*
    E_prime = lambda x: E_rot(x) - xPrime
    H_prime = lambda x: H_rot(x) - xPrime
*/

array<double, 3> translate_vector(const array<double, 3>& v, const array<double, 3>& xPrime) {
    return {{
        v[0] - xPrime[0],
        v[1] - xPrime[1],
        v[2] - xPrime[2]
    }};
}

// =========================================
//  Function to Move the Dipole Field
// =========================================
/*
    return E_prime, H_prime
*/

pair<array<double, 3>, array<double, 3>> moveDipoleField(
    const array<double, 3>& xPrime,
    const array<double, 3>& rPrime,
    const array<double, 3>& E,
    const array<double, 3>& H
) {
    // Compute rotation matrices
    auto Ry = rotation_matrix_y(rPrime);
    auto Rz = rotation_matrix_z(rPrime);

    // Rotate E and H fields
    auto E_rot = rotate_vector(E, Ry);
    E_rot = rotate_vector(E_rot, Rz);

    auto H_rot = rotate_vector(H, Ry);
    H_rot = rotate_vector(H_rot, Rz);

    // Translate the rotated fields
    auto E_prime = translate_vector(E_rot, xPrime);
    auto H_prime = translate_vector(H_rot, xPrime);

    return {E_prime, H_prime};
}

// =========================================
//  Main Function for Testing
// =========================================
int main() {
    // Define test inputs
    array<double, 3> xPrime = {1.0, 1.0, 1.0};
    array<double, 3> rPrime = {0.0, 1.0, 1.0};
    array<double, 3> E = {2.0, -1.0, 3.0};
    array<double, 3> H = {1.0, 2.0, -1.0};

    // Compute moved dipole fields
    auto [E_new, H_new] = moveDipoleField(xPrime, rPrime, E, H);

    // Print results
    cout << "E_prime: (" << E_new[0] << ", " << E_new[1] << ", " << E_new[2] << ")" << endl;
    cout << "H_prime: (" << H_new[0] << ", " << H_new[1] << ", " << H_new[2] << ")" << endl;

    return 0;
}
