#include <Eigen/Dense>
#include <iostream>
#include <complex>
#include <fstream>

#include "Constants.h"

using namespace Eigen;
using namespace std;

const complex<double> j(0, 1);  // Imaginary unit

struct Constants constants;

const double omega = 1.0;
const double epsilon0 = 1.0; //constants.epsilon0;
const double n0 = constants.n0;
const double mu0 = 1.0; //constants.mu0;
const double k0 = 1.0;// constants.k0;

const double epsilon_air = epsilon0;
const double epsilon_substrate = 2.0; // ?????????????
const double n_air = n0;
const double n_substrate = sqrt(epsilon_substrate/epsilon_air);//constants.n1;
const double k_air = k0;
const double k_substrate = constants.k1;

const double eta_air = 1.0; //constants.eta0; //sqrt(omega/epsilon0);
const double eta_substrate = sqrt(mu0/epsilon_substrate);

// Utility function to save MatrixXcd to CSV
void saveMatrixCSV(const std::string& filename, const Eigen::MatrixXcd& mat) {
    std::ofstream file(filename);
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            file << mat(i, j).real() << "+" << mat(i, j).imag() << "i";
            if (j < mat.cols() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

// Utility function to save VectorXcd to CSV
void saveVectorCSV(const std::string& filename, const Eigen::VectorXcd& vec) {
    std::ofstream file(filename);
    for (int i = 0; i < vec.size(); ++i) {
        file << vec(i).real() << "+" << vec(i).imag() << "i\n";
    }
    file.close();
}

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

    E_r_out = eta_air * Iel * cos_theta / (2.0 * constants.pi * r * r) 
              * (1.0 + 1.0 / (j * k0 * r)) * expK0r;
    
    E_theta_out = (j * eta_air * Iel * sin_theta / (4.0 * constants.pi * r)) 
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


Eigen::Matrix3d rotation_matrix_y_inv(const Eigen::Vector3d& rPrime) {
    double cosT = cos_theta(rPrime);
    double sinT = sin_theta(rPrime);
    Eigen::Matrix3d Ry {{ cosT,  0, -sinT },
                        { 0,     1,   0  },
                        { sinT,  0,  cosT }
                        };
    return Ry;
}


Eigen::Matrix3d rotation_matrix_z_inv(const Eigen::Vector3d& rPrime) {
    double cosP = cos_phi(rPrime);
    double sinP = sin_phi(rPrime);
    Eigen::Matrix3d Rz {{cosP,  sinP, 0} ,
                        {-sinP, cosP, 0} ,
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

pair<Eigen::Vector3cd, Eigen::Vector3cd> moveDipoleField(
    const Eigen::Vector3d& xPrime,
    const Eigen::Vector3d& rPrime,
    const Eigen::Vector3cd& E,
    const Eigen::Vector3cd& H
) {
    // Compute rotation matrices
    auto Ry_inv = rotation_matrix_y_inv(rPrime);
    auto Rz_inv = rotation_matrix_z_inv(rPrime);

    // Rotate the field components
    Vector3cd E_rot = Ry_inv * Rz_inv * E;
    Vector3cd H_rot = Ry_inv * Rz_inv * H;

    // Translate the fields
    Vector3cd E_prime = E_rot - xPrime;
    Vector3cd H_prime = H_rot - xPrime;

    return {E_prime, H_prime};
}



pair<Eigen::Vector3cd, Eigen::Vector3cd> hertzianDipoleField(const Eigen::Vector3d& x,      // point to evaluate field in
    const double Iel,               // "size" of Hertzian dipole
    const Eigen::Vector3d& xPrime,  // Hertzian dipole placement
    const Eigen::Vector3d& rPrime,  // Hertzian dipole direction
    const double k0                 // wave constant
    )
{
    auto Ry = rotation_matrix_y(rPrime);
    auto Rz = rotation_matrix_z(rPrime);

    Eigen::Vector3d x_origo = Rz * Ry * (x - xPrime);
    double r = x_origo.norm();
    double cos_theta = 0.0, sin_theta = 0.0, cos_phi = 0.0, sin_phi = 0.0;

    computeAngles(x_origo, r, cos_theta, sin_theta, cos_phi, sin_phi);

    // Compute exp(-j * k0 * r) once
    complex<double> expK0r = polar(1.0, -k0 * r);

    auto [E_origo, H_origo] = fieldDipole(r, Iel, cos_theta, sin_theta, cos_phi, sin_phi, expK0r);

    auto [E, H] = moveDipoleField(xPrime, rPrime, E_origo, H_origo);

    return {E, H};
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

pair<Eigen::Vector3cd, Eigen::Vector3cd> reflected_field_TE(double Gamma_r, double cos_theta_inc, double sin_theta_inc, const Eigen::Vector3d& x,
                                    double E0, double k1 = k_air, double eta0 = eta_air) {
    complex<double> wave = Gamma_r * E0 * exp(-j * k1 * (sin_theta_inc * x[0] - cos_theta_inc * x[2]));
    
    Eigen::Vector3cd E(0.0, wave, 0.0);
    Eigen::Vector3cd H(cos_theta_inc * wave / eta0, 0.0, sin_theta_inc * wave / eta0);

    return {E, H};
}

pair<Eigen::Vector3cd, Eigen::Vector3cd> reflected_field_TM(double Gamma_r, double cos_theta_inc, double sin_theta_inc, const Eigen::Vector3d& x, 
                                    double E0, double k1 = k_air, double eta0 = eta_air) {
    complex<double> wave = E0 * Gamma_r * exp(-j * k1 * (x[0] * sin_theta_inc - x[2] * cos_theta_inc));
    
    Eigen::Vector3cd E (cos_theta_inc * wave, 0.0, sin_theta_inc * wave);
    Eigen::Vector3cd H (0.0, - wave/eta0, 0);
    return {E, H};
}


pair<Eigen::Vector3cd, Eigen::Vector3cd> transmitted_field_TE(double Gamma_t, double sin_theta_inc, const Eigen::Vector3d& x, double E0, 
                                      double k1 = k_air, double k2 = k_substrate, double eta1 = eta_substrate) {
    double sin_theta_trans = k1 / k2 * sin_theta_inc;
    double cos_theta_trans = sqrt(1 - sin_theta_trans * sin_theta_trans);

    complex<double> wave = Gamma_t * E0 * exp(-j * k2 * (sin_theta_trans * x[0] + cos_theta_trans * x[2]));

    Eigen::Vector3cd E(0.0, wave , 0.0);
    Eigen::Vector3cd H(-cos_theta_trans * wave / eta1, 0.0 , sin_theta_trans * wave / eta1);
    
    return {E, H};
}

pair<Eigen::Vector3cd, Eigen::Vector3cd> transmitted_field_TM(double Gamma_t, double sin_theta_inc, const Eigen::Vector3d& x, double E0, 
                                      double k1 = k_air, double k2 = k_substrate, double eta1 = eta_substrate) {
    double sin_theta_trans = k1 / k2 * sin_theta_inc;
    double cos_theta_trans = sqrt(1 - sin_theta_trans * sin_theta_trans);

    complex<double> wave = E0 * Gamma_t * exp(-j * k2 * (x[0] * sin_theta_trans + x[2] * cos_theta_trans));

    Eigen::Vector3cd E (wave * cos_theta_trans, 0.0, - wave * sin_theta_trans);
    Eigen::Vector3cd H (0.0, wave / eta1, 0.0);

    return {E, H};
}

// =========================================
//  Driver Function
// =========================================

pair<Eigen::Vector3cd, Eigen::Vector3cd> fieldIncidentNew(const Eigen::Vector3d& k_inc, const Eigen::Vector3cd& E_inc, const Eigen::Vector3d& x) {
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

    auto [E_ref_TE, H_ref_TE] = reflected_field_TE(Gamma_r_TE, cos_theta_inc, sin_theta_inc, x, E0);
    auto [E_ref_TM, H_ref_TM] = reflected_field_TM(Gamma_r_TM, cos_theta_inc, sin_theta_inc, x, E0);
    auto [E_trans_TE, H_trans_TE] = transmitted_field_TE(Gamma_t_TE, sin_theta_inc, x, E0);
    auto [E_trans_TM, H_trans_TM] = transmitted_field_TM(Gamma_t_TM, sin_theta_inc, x, E0);

    auto Rz_inv = rotation_matrix_z_inv(cosP, sinP);

    Eigen::Vector3cd E_tot_1 = E_inc + Rz_inv * (cos_beta * E_ref_TE + sin_beta * E_ref_TM);
    Eigen::Vector3cd E_tot_2 = Rz_inv * (cos_beta * E_trans_TE + sin_beta * E_trans_TM);

    Eigen::Vector3cd H_tot_1 = E_inc + Rz_inv * (cos_beta * H_ref_TE + sin_beta * H_ref_TM);
    Eigen::Vector3cd H_tot_2 = Rz_inv * (cos_beta * H_trans_TE + sin_beta * H_trans_TM);

    Eigen::Vector3cd E_tot = (x[2] < 0) ? E_tot_2 : E_tot_1;
    Eigen::Vector3cd H_tot = (x[2] < 0) ? H_tot_2 : H_tot_1;

    return {E_tot, H_tot}; 
}

pair<Eigen::MatrixXcd, Eigen::VectorXcd> linearSystem(const Eigen::MatrixX3d& x_mu, const Eigen::MatrixX3d& x_nu_prime, 
    const Eigen::MatrixX3d& x_nu_prime_tilde, const Eigen::MatrixX3d& x_nu_2prime, 
    const Eigen::MatrixX3d& tau1_mat, const Eigen::MatrixX3d& tau2_mat, 
    const Eigen::MatrixX3d& dipole1_int, const Eigen::MatrixX3d& dipole2_int,
    const Eigen::MatrixX3d& dipole1_int_tilde, const Eigen::MatrixX3d& dipole2_int_tilde,  
    const Eigen::MatrixX3d& dipole1_ext, const Eigen::MatrixX3d& dipole2_ext, 
    const double Iel, const double k0, const double Gamma_r,
    const Eigen::Vector3d& k_inc, const Eigen::Vector3cd& E_inc, const Eigen::Vector3cd& H_inc){


int M = x_mu.rows();
int Nprime = x_nu_prime.rows();
int N2prime = x_nu_2prime.rows();


Eigen::MatrixXcd A(4*M, 2*Nprime + 2*N2prime);
Eigen::VectorXcd b(4*M);

for (int mu = 0; mu < M; mu++){

Eigen::Vector3d x_mu_it = x_mu.row(mu);
Eigen::Vector3d tau1 = tau1_mat.row(mu);
Eigen::Vector3d tau2 = tau2_mat.row(mu);

/////////////////////////

// BUILDING RIGHT HAND SIDE

auto [E_incnew, H_incnew] = fieldIncidentNew(k_inc, E_inc, x_mu_it);

b(mu) = - E_inc.dot(tau1);
b(mu + M) = - E_inc.dot(tau2);
b(mu + 2*M) = - H_inc.dot(tau1);
b(mu + 3*M) = - H_inc.dot(tau2);

/////////////////////////

// BUILDING SYSTEM MATRIX (AVOIDING EXTRA LOOPS)

for (int nu = 0; nu < Nprime; nu++){

// Save relevant vectors as vectors, to use functions correctly
Eigen::Vector3d x_nu_it = x_nu_prime.row(nu);
Eigen::Vector3d x_nu_tilde_it = x_nu_prime_tilde.row(nu);

Eigen::Vector3d dipole1 = dipole1_int.row(nu);
Eigen::Vector3d dipole2 = dipole2_int.row(nu);

Eigen::Vector3d dipole1_tilde = dipole1_int_tilde.row(nu);
Eigen::Vector3d dipole2_tilde = dipole2_int_tilde.row(nu);

/////////////////////////

// FIELDS NEEDED
// Dipole 1
auto [E1_nuprime, H1_nuprime] = hertzianDipoleField(x_mu_it, Iel, x_nu_it, dipole1, k0);
auto [E1_nuprime_tilde, H1_nuprime_tilde] = hertzianDipoleField(x_mu_it, Iel, x_nu_tilde_it, dipole1_tilde, k0);

// Dipole 2
auto [E2_nuprime, H2_nuprime] = hertzianDipoleField(x_mu_it, Iel, x_nu_it, dipole2, k0);
auto [E2_nuprime_tilde, H2_nuprime_tilde] = hertzianDipoleField(x_mu_it, Iel, x_nu_tilde_it, dipole2_tilde, k0);

/////////////////////////

// Electric fields
// A(1,1) 
A(mu, nu) = E1_nuprime.dot(tau1) ;

// A(1,2)
A(mu, nu + Nprime) = E2_nuprime.dot(tau1) ;

// A(2,1)
A(mu + M, nu) = E1_nuprime.dot(tau2) ;

// A(2,2)
A(mu + M, nu + Nprime) = E2_nuprime.dot(tau2) ;

// Magnetic fields
// A(3,1)
A(mu + 2*M, nu) = H1_nuprime.dot(tau1);

// A(3,2)
A(mu + 2*M, nu + Nprime) = H2_nuprime.dot(tau1) ;

// A(4,1)
A(mu + 3*M, nu) = H1_nuprime.dot(tau2);

// A(4,2)
A(mu + 3*M, nu + Nprime) = H2_nuprime.dot(tau2);
}

for (int nu = 0; nu < N2prime; nu++){

// Save relevant vectors as vectors, to use functions correctly
Eigen::Vector3d x_nu_it = x_nu_2prime.row(nu);

Eigen::Vector3d dipole1 = dipole1_ext.row(nu);
Eigen::Vector3d dipole2 = dipole2_ext.row(nu);

/////////////////////////

// FIELDS NEEDED
// Dipole 1
auto [E1_nuprime, H1_nuprime] = hertzianDipoleField(x_mu_it, Iel, x_nu_it, dipole1, k0);

// Dipole 2
auto [E2_nuprime, H2_nuprime] = hertzianDipoleField(x_mu_it, Iel, x_nu_it, dipole2, k0);

/////////////////////////

// Electric fields
// A(1,3) 
A(mu, nu + 2*Nprime) = E1_nuprime.dot(tau1);

// A(1,4)
A(mu, nu + 2*Nprime + N2prime) = E2_nuprime.dot(tau1);

// A(2,3)
A(mu + M, nu + 2*Nprime) = E1_nuprime.dot(tau2);

// A(2,4)
A(mu + M, nu + 2*Nprime + N2prime)  = E2_nuprime.dot(tau2);

// Magnetic fields
// A(3,3)
A(mu + 2*M, nu + 2*Nprime) = H1_nuprime.dot(tau1);

// A(3,4)
A(mu + 2*M, nu + 2*Nprime + N2prime) = H2_nuprime.dot(tau1);

// A(4,3)
A(mu + 3*M, nu + 2*Nprime) = H1_nuprime.dot(tau2);

// A(4,4)
A(mu + 3*M, nu + 2*Nprime + N2prime) = H2_nuprime.dot(tau2);
}
}

return {A,b};

}

void sphere(
    double radius,
    const Vector3d& center,
    int num_points,
    MatrixXd& points,
    MatrixXd& normals,
    MatrixXd& tau1,
    MatrixXd& tau2
) {
    int N = num_points;

    // Create theta and phi vectors (excluding first and last element as in Python)
    VectorXd theta0 = VectorXd::LinSpaced(N + 2, 0.0, constants.pi).segment(1, N);
    VectorXd phi0 = VectorXd::LinSpaced(N + 2, 0.0, 2 * constants.pi).segment(1, N);

    // Meshgrid equivalent
    MatrixXd theta(N, N), phi(N, N);
    for (int i = 0; i < N; ++i) {
        theta.row(i) = theta0.transpose();
        phi.col(i) = phi0;
    }

    ArrayXXd st = theta.array().sin();
    ArrayXXd ct = theta.array().cos();
    ArrayXXd sp = phi.array().sin();
    ArrayXXd cp = phi.array().cos();

    ArrayXXd x = center(0) + radius * st * cp;
    ArrayXXd y = center(1) + radius * st * sp;
    ArrayXXd z = center(2) + radius * ct;

    // Might be redundant
    int total_points = N * N;
    points.resize(total_points, 3);
    normals.resize(total_points, 3);
    tau1.resize(total_points, 3);
    tau2.resize(total_points, 3);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;

            Vector3d p(x(i, j), y(i, j), z(i, j));
            points.row(idx) = p;

            // Normal vector
            Vector3d n = (p - center).normalized();
            normals.row(idx) = n;

            // tau1 vector
            Vector3d t1(
                radius * ct(i, j) * cp(i, j),
                radius * ct(i, j) * sp(i, j),
                -radius * st(i, j)
            );
            tau1.row(idx) = t1.normalized();

            // tau2 vector
            Vector3d t2(
                -radius * st(i, j) * sp(i, j),
                radius * st(i, j) * cp(i, j),
                0.0
            );
            tau2.row(idx) = t2.normalized();
        }
    }
}

int main() {
    using namespace Eigen;

    int num_points = 5;
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
    Vector3cd H_inc;
    H_inc << 0.0, 1.0, 0.0;

    // Call the linear system
    auto [A, b] = linearSystem(points_mu, points_nu_prime, points_nu_prime_tilde,
                               points_nu_2prime, tau1_mu, tau2_mu,
                               dipole1_int, dipole2_int,
                               dipole1_int_tilde, dipole2_int_tilde,
                               dipole1_ext, dipole2_ext,
                               Iel, k0, Gamma_r, k_inc, E_inc, H_inc);

    std::cout << "Linear system A has size: " << A.rows() << " x " << A.cols() << std::endl;
    std::cout << "Right-hand side b has size: " << b.size() << std::endl;

    VectorXcd y = A.fullPivLu().solve(b);
    std::cout << "Solution x has size: " << y.size() << std::endl;

    

    // At the end of main()
    saveMatrixCSV("matrix_A_simple.csv", A);
    saveVectorCSV("vector_b_simple.csv", b);
    saveVectorCSV("solution_y_simple.csv", y);


    return 0;
}
