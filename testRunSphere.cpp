#include <Eigen/Dense>
#include <iostream>
#include <complex>

#include "Constants.h"

using namespace Eigen;
using namespace std;

const complex<double> j(0, 1);  // Imaginary unit

struct Constants constants;

const double omega = 1.0;
const double epsilon0 = constants.epsilon0;
const double n0 = constants.n0;
const double mu0 = constants.mu0;
const double k0 = constants.k0;

const double epsilon_air = epsilon0;
const double epsilon_substrate = 2.0; // ?????????????
const double n_air = n0;
const double n_substrate = constants.n1;
const double k_air = k0;
const double k_substrate = constants.k1;

const double eta_air = constants.eta0; //sqrt(omega/epsilon0);
const double eta_substrate = sqrt(mu0/epsilon_substrate);




int main() {
    using namespace Eigen;

    int num_points = 10;
    Vector3d center(0.0, 0.0, 1.0);

    // Sphere 1: sphere_mu
    MatrixXd points_mu, normals_mu, tau1_mu, tau2_mu;
    sphere(1.0, center, num_points, points_mu, normals_mu, tau1_mu, tau2_mu);

    // Sphere 2: sphere_nu_prime
    MatrixXd points_nu_prime, normals_nu_prime, tau1_nu_prime, tau2_nu_prime;
    sphere(0.8, center, num_points, points_nu_prime, normals_nu_prime, tau1_nu_prime, tau2_nu_prime);

    // Sphere 3: sphere_nu_prime_tilde (mirror image in z -> -z)
    MatrixXd points_nu_prime_tilde = points_nu_prime;
    points_nu_prime_tilde.col(2) *= -1;
    MatrixXd normals_nu_prime_tilde = normals_nu_prime;
    normals_nu_prime_tilde.col(2) *= -1;
    MatrixXd tau1_nu_prime_tilde = tau1_nu_prime;
    tau1_nu_prime_tilde.col(2) *= -1;
    MatrixXd tau2_nu_prime_tilde = tau2_nu_prime;
    tau2_nu_prime_tilde.col(2) *= -1;

    // Sphere 4: sphere_nu_2prime
    MatrixXd points_nu_2prime, normals_nu_2prime, tau1_nu_2prime, tau2_nu_2prime;
    sphere(1.2, center, num_points, points_nu_2prime, normals_nu_2prime, tau1_nu_2prime, tau2_nu_2prime);

    // Assume dipole directions are tau1 and tau2 for all
    auto dipole1_int = tau1_nu_prime;
    auto dipole2_int = tau2_nu_prime;
    auto dipole1_int_tilde = tau1_nu_prime_tilde;
    auto dipole2_int_tilde = tau2_nu_prime_tilde;
    auto dipole1_ext = tau1_nu_2prime;
    auto dipole2_ext = tau2_nu_2prime;

    // Parameters for the linear system
    double Iel = 1.0;
    double k0 = 2 * constants.pi; // Example wavenumber
    double Gamma_r = 1.0; // Reflection coefficient
    Vector3d k_inc(0.0, 0.0, 1.0);
    Vector3cd E_inc;
    E_inc << 1.0, 0.0, 0.0;

    // Call the linear system
    auto [A, b] = linearSystem(points_mu, points_nu_prime, points_nu_prime_tilde,
                               points_nu_2prime, tau1_mu, tau2_mu,
                               dipole1_int, dipole2_int,
                               dipole1_int_tilde, dipole2_int_tilde,
                               dipole1_ext, dipole2_ext,
                               Iel, k0, Gamma_r, k_inc, E_inc);

    std::cout << "Linear system A has size: " << A.rows() << " x " << A.cols() << std::endl;
    std::cout << "Right-hand side b has size: " << b.size() << std::endl;

    return 0;
}
