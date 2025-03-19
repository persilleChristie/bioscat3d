#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

//g++ moveDipole.cpp -o mD.exe -I"D:/vcpkg/installed/x64-windows/include/eigen3"

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

inline double norm(const Eigen::Vector3d& x) {
    return std::hypot(x[0], x[1], x[2]);
}

inline double cos_theta(const Eigen::Vector3d& rPrime) {
    return rPrime[2] / norm(rPrime);
}

inline double sin_theta(const Eigen::Vector3d& rPrime) {
    double cosT = cos_theta(rPrime);
    cosT = std::clamp(cosT, -1.0, 1.0);
    return sqrt(1 - cosT * cosT);
}

/*
    cos_phi = np.sign(rPrime[1]) * rPrime[0] / np.sqrt(rPrime[0]**2 + rPrime[1]**2)
    sin_phi = np.sqrt(1 - cos_phi**2)
*/

inline double cos_phi(const Eigen::Vector3d& rPrime) {
    double xy_norm = std::hypot(rPrime[0], rPrime[1]);
    return (xy_norm == 0) ? 1.0 : rPrime[0] / xy_norm;
}

inline double sin_phi(const Eigen::Vector3d& rPrime) {
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

Eigen::Matrix3d rotation_matrix_y(const Eigen::Vector3d& rPrime) {
    double cosT = cos_theta(rPrime);
    double sinT = sin_theta(rPrime);
    Eigen::Matrix3d Ry {{ cosT,  0, sinT },
                        { 0,     1,   0  },
                        { -sinT, 0, cosT }
                        };
    return Ry;
}

/*
    # rotation matrix for phi around z axis
    Rz = np.array([[cos_phi, -sin_phi, 0],
                   [sin_phi, cos_phi, 0],
                   [0, 0, 1]])
*/

Eigen::Matrix3d rotation_matrix_z(const Eigen::Vector3d& rPrime) {
    double cosP = cos_phi(rPrime);
    double sinP = sin_phi(rPrime);
    Eigen::Matrix3d Rz {{cosP, -sinP, 0} ,
                        {sinP,  cosP, 0} ,
                        {0,      0,   1}
                        };
    return Rz;
}

// =========================================
//  Rotate the Field Components
// =========================================
/*
    # rotate the field
    E_rot = lambda x: np.dot(Rz, np.dot(Ry, E(x)))
    H_rot = lambda x: np.dot(Rz, np.dot(Ry, H(x)))
*/

// =========================================
//  Compute Final Translated Fields
// =========================================
/*
    E_prime = lambda x: E_rot(x) - xPrime
    H_prime = lambda x: H_rot(x) - xPrime
*/

Eigen::Vector3d translate_vector(const Eigen::Vector3d& v, const Eigen::Vector3d& xPrime) {
    return v - xPrime;
}

// =========================================
//  Function to Move the Dipole Field
// =========================================
/*
    return E_prime, H_prime
*/

pair<Eigen::Vector3d, Eigen::Vector3d> moveDipoleField(
    const Eigen::Vector3d& xPrime,
    const Eigen::Vector3d& rPrime,
    const Eigen::Vector3d& E,
    const Eigen::Vector3d& H
) {
    // Compute rotation matrices
    auto Ry = rotation_matrix_y(rPrime);
    auto Rz = rotation_matrix_z(rPrime);

    // Rotate the field components
    auto E_rot = Rz * Ry * E;
    auto H_rot = Rz * Ry * H;

    // Translate the fields
    auto E_prime = E_rot - xPrime;
    auto H_prime = H_rot - xPrime;

    return {E_prime, H_prime};
}

// =========================================
//  Main Function for Testing
// =========================================
int main() {
    // Define test inputs
    Eigen::Vector3d xPrime(1.0, 1.0, 1.0);
    Eigen::Vector3d rPrime(0.0, 1.0, 1.0);
    Eigen::Vector3d E(2.0, -1.0, 3.0);
    Eigen::Vector3d H(1.0, 2.0, -1.0);

    // Compute moved dipole fields
    auto [E_new, H_new] = moveDipoleField(xPrime, rPrime, E, H);

    // Print results
    cout << "E_prime: (" << E_new[0] << ", " << E_new[1] << ", " << E_new[2] << ")" << endl;
    cout << "H_prime: (" << H_new[0] << ", " << H_new[1] << ", " << H_new[2] << ")" << endl;

    return 0;
}
