#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <cmath>

#include "moveDipoleField.h"
#include "fieldDipole.h"
#include "eIncidentNew.h"
#include "computeAngles.h"

using namespace std;


// ?? TODO: TEMPLATES ??

Eigen::MatrixXcd systemMatrix(const Eigen::MatrixX3d& x_mu, const Eigen::MatrixX3d& x_nu_prime, 
                                const Eigen::MatrixX3d& x_nu_prime_tilde, const Eigen::MatrixX3d& x_nu_2prime, 
                                const int Nprime, const int N2prime, const int M,
                                const Eigen::MatrixX3d& rPrime, const double Iel, double k0){
    
    Eigen::MatrixXcd A;

    for (int i = 0; i < Nprime; i++){
        for (int j = 0; j < M; j++){

            Eigen::Vector3d xPrime = x_mu.row(j);

            double r = xPrime.norm();
            double cos_theta = 0.0, sin_theta = 0.0, cos_phi = 0.0, sin_phi = 0.0;
            
            computeAngles(xPrime, r, cos_theta, sin_theta, cos_phi, sin_phi);
            
            // Compute exp(-j * k0 * r) once
            complex<double> expK0r = std::polar(1.0, -k0 * r);

            auto [E1_origo, H1_origo] = fieldDipole(r, Iel, cos_theta, sin_theta, cos_phi, sin_phi, expK0r);
        }
    }

    for (int i = 0; i < N2prime; i++){

    }


};