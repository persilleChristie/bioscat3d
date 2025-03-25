#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <cmath>

#include "hertzianDipoleField.h"
#include "fieldIncidentNew.h"
#include "computeAngles.h"

using namespace std;


// ?? TODO: TEMPLATES ??

pair<Eigen::MatrixXcd, Eigen::VectorXcd> linearSystem(const Eigen::MatrixX3d& x_mu, const Eigen::MatrixX3d& x_nu_prime, 
                                const Eigen::MatrixX3d& x_nu_prime_tilde, const Eigen::MatrixX3d& x_nu_2prime, 
                                const Eigen::MatrixX3d& tau1_mat, const Eigen::MatrixX3d& tau2_mat, 
                                const Eigen::MatrixX3d& dipole1_int, const Eigen::MatrixX3d& dipole2_int, 
                                const Eigen::MatrixX3d& dipole1_ext, const Eigen::MatrixX3d& dipole2_ext, 
                                const double Iel, const double k0, const double Gamma_r,
                                const Eigen::Vector3d& k_inc, const Eigen::Vector3cd& E_inc){
    
    Eigen::MatrixXcd A;
    Eigen::VectorXcd b;

    int M = static_cast<int>(x_mu.size());
    int Nprime = static_cast<int>(x_nu_prime.size());
    int N2prime = static_cast<int>(x_nu_2prime.size());

    for (int mu = 0; mu < M; mu++){

        Eigen::Vector3d x_mu_it = x_mu.row(mu);
        Eigen::Vector3d tau1 = tau1_mat.row(mu);
        Eigen::Vector3d tau2 = tau2_mat.row(mu);

        /////////////////////////

        // BUILDING RIGHT HAND SIDE

        auto [E_incnew, H_incnew] = fieldIncidentNew(k_inc, E_inc, x_mu_it);

        b[mu] = - E_incnew.dot(tau1);
        b[mu + M] = - E_incnew.dot(tau2);
        b[mu + 2*M] = - H_incnew.dot(tau1);
        b[mu + 3*M] = - H_incnew.dot(tau2);

        /////////////////////////

        // BUILDING SYSTEM MATRIX (AVOIDING EXTRA LOOPS)

        for (int nu = 0; nu < Nprime; nu++){

            // Save relevant vectors as vectors, to use functions correctly
            Eigen::Vector3d x_nu_it = x_nu_prime.row(nu);
            Eigen::Vector3d x_nu_tilde_it = x_nu_prime_tilde.row(nu);

            Eigen::Vector3d dipole1 = dipole1_int.row(nu);
            Eigen::Vector3d dipole2 = dipole2_int.row(nu);
            
            /////////////////////////

            // FIELDS NEEDED
            // Dipole 1
            auto [E1_nuprime, H1_nuprime] = hertzianDipoleField(x_mu_it, Iel, x_nu_it, dipole1, k0);
            auto [E1_nuprime_tilde, H1_nuprime_tilde] = hertzianDipoleField(x_mu_it, Iel, x_nu_tilde_it, dipole1, k0);

            // Dipole 2
            auto [E2_nuprime, H2_nuprime] = hertzianDipoleField(x_mu_it, Iel, x_nu_it, dipole2, k0);
            auto [E2_nuprime_tilde, H2_nuprime_tilde] = hertzianDipoleField(x_mu_it, Iel, x_nu_tilde_it, dipole2, k0);

            /////////////////////////

            // Electric fields
            // A(1,1) 
            A[mu, nu] = E1_nuprime.dot(tau1) + Gamma_r * E1_nuprime_tilde.dot(tau1);
            
            // A(1,2)
            A[mu, nu + Nprime] = E2_nuprime.dot(tau1) + Gamma_r * E2_nuprime_tilde.dot(tau1);

            // A(2,1)
            A[mu + M, nu] = E1_nuprime.dot(tau2) + Gamma_r * E1_nuprime_tilde.dot(tau2);
            
            // A(2,2)
            A[mu + M, nu + Nprime] = E2_nuprime.dot(tau2) + Gamma_r * E2_nuprime_tilde.dot(tau2);

            // Magnetic fields
            // A(3,1)
            A[mu + 2*M, nu] = H1_nuprime.dot(tau1) + Gamma_r * H1_nuprime_tilde.dot(tau1);
            
            // A(3,2)
            A[mu + 2*M, nu + Nprime] = H2_nuprime.dot(tau1) + Gamma_r * H2_nuprime_tilde.dot(tau1);

            // A(4,1)
            A[mu + 3*M, nu] = H1_nuprime.dot(tau2) + Gamma_r * H1_nuprime_tilde.dot(tau2);
            
            // A(4,2)
            A[mu + 3*M, nu + Nprime] = H2_nuprime.dot(tau2) + Gamma_r * H2_nuprime_tilde.dot(tau2);
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
            A[mu, nu + 2*Nprime] = E1_nuprime.dot(tau1);
            
            // A(1,4)
            A[mu, nu + 2*Nprime + N2prime] = E2_nuprime.dot(tau1);

            // A(2,3)
            A[mu + M, nu + 2*Nprime] = E1_nuprime.dot(tau2);
            
            // A(2,4)
            A[mu + M, nu + 2*Nprime + N2prime]  = E2_nuprime.dot(tau2);

            // Magnetic fields
            // A(3,3)
            A[mu + 2*M, nu + 2*Nprime] = H1_nuprime.dot(tau1);
            
            // A(3,4)
            A[mu + 2*M, nu + 2*Nprime + N2prime] = H2_nuprime.dot(tau1);

            // A(4,3)
            A[mu + 3*M, nu + 2*Nprime] = H1_nuprime.dot(tau2);
            
            // A(4,4)
            A[mu + 3*M, nu + 2*Nprime + N2prime] = H2_nuprime.dot(tau2);
        }
    }

    return {A,b};

}