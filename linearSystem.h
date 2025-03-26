#ifndef _LINEAR_SYSTEM_H
#define _LINEAR_SYSTEM_H

#include <iostream>
#include <Eigen/Dense>

using namespace std;


pair<Eigen::MatrixXcd, Eigen::VectorXcd> linearSystem(const Eigen::MatrixX3d& x_mu, const Eigen::MatrixX3d& x_nu_prime, 
    const Eigen::MatrixX3d& x_nu_prime_tilde, const Eigen::MatrixX3d& x_nu_2prime, 
    const Eigen::MatrixX3d& tau1_mat, const Eigen::MatrixX3d& tau2_mat, 
    const Eigen::MatrixX3d& dipole1_int, const Eigen::MatrixX3d& dipole2_int,
    const Eigen::MatrixX3d& dipole1_int_tilde, const Eigen::MatrixX3d& dipole2_int_tilde,  
    const Eigen::MatrixX3d& dipole1_ext, const Eigen::MatrixX3d& dipole2_ext, 
    const double Iel, const double k0, const double Gamma_r,
    const Eigen::Vector3d& k_inc, const Eigen::Vector3cd& E_inc);


#endif