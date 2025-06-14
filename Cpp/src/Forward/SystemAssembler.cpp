#include <iostream>
#include <memory>
#include "../../lib/Forward/SystemAssembler.h"
#include "../../lib/Forward/FieldCalculator.h"
#include "../../lib/Forward/FieldCalculatorDipole.h"
#include "../../lib/Forward/FieldCalculatorUPW.h"
#include "../../lib/Utils/UtilsSolvers.h"
#include "../../lib/Utils/Constants.h"

using namespace Eigen;

/// @file SystemAssembler.cpp
/// @brief Implementation of the SystemAssembler class, which assembles the system matrix and right-hand side vector for the MAS.
/// @details This class constructs the system matrix `A` and the right-hand side vector `b` based on the contributions from interior and exterior sources, as well as the incident field.
/// @note The system is assembled based on the evaluation points, tangential vectors, and sources provided.
/// @note The method `assembleSystem` computes the contributions from both interior and exterior sources, as well as the incident field, to populate the system matrix and right-hand side vector.
/// @note The method assumes that the sources are provided in pairs, where each pair corresponds to a dipole at the same point.
void SystemAssembler::assembleSystem(
    Eigen::MatrixXcd& A,
    Eigen::VectorXcd& b,
    Eigen::MatrixX3d& points,
    Eigen::MatrixX3d& tau1,
    Eigen::MatrixX3d& tau2,
    const std::vector<std::shared_ptr<FieldCalculator>>& sources_int,
    const std::vector<std::shared_ptr<FieldCalculator>>& sources_ext,
    const std::shared_ptr<FieldCalculator>& incident
) {

    int M = points.rows();

    if (M == 0) {
        std::cerr << "[Error] Surface has no points!\n";
        return;
    }
    if (tau1.rows() != M || tau2.rows() != M) {
        std::cerr << "[Error] Tangents and points mismatch!\n";
        return;
    }
    
    int Nprime_double = sources_int.size();
    int N2prime_double = sources_ext.size();
    int Nprime = Nprime_double/2;
    int N2prime = N2prime_double/2;

    A.resize(4 * M, Nprime_double + N2prime_double);
    b.resize(4 * M);

    MatrixX3cd E_inc_new(M,3);
    MatrixX3cd H_inc_new(M,3);

    incident->computeFields(E_inc_new, H_inc_new, points);

    // Allocate space for placeholders
    MatrixX3cd E_HD(1,3);
    MatrixX3cd H_HD(1,3);

    for (int mu = 0; mu < M; ++mu) {
        MatrixX3d x_mu = points.row(mu);
        Vector3d t1 = tau1.row(mu);
        Vector3d t2 = tau2.row(mu);
        
        b(mu)       = -t1.dot(E_inc_new.row(mu));
        b(mu + M)   = -t2.dot(E_inc_new.row(mu));
        b(mu + 2*M) = -t1.dot(H_inc_new.row(mu));
        b(mu + 3*M) = -t2.dot(H_inc_new.row(mu));
        

        for (int nu = 0; nu < Nprime; ++nu) {
            // It is assumed that dipoles lie in pairs, i.e. 2*nu and 2*nu + 1 are for the same point
            sources_int[2*nu]->computeFields(E_HD, H_HD, x_mu);

            Vector3cd E_int1 = E_HD.row(0);
            Vector3cd H_int1 = H_HD.row(0);

        
            sources_int[2*nu + 1]->computeFields(E_HD, H_HD, x_mu);

            Vector3cd E_int2 = E_HD.row(0);
            Vector3cd H_int2 = H_HD.row(0);
            

            // Electric fields
            // A(1,1) 
            A(mu, nu) = t1.transpose() * E_int1; 
            
            // A(1,2)
            A(mu, nu + Nprime) = t1.transpose() * E_int2; 

            // A(2,1)
            A(mu + M, nu) = t2.transpose() * E_int1; 

            // A(2,2)
            A(mu + M, nu + Nprime) = t2.transpose() * E_int2;
            // Magnetic fields
            // A(3,1)
            A(mu + 2*M, nu) = t1.transpose() * H_int1; 
            
            // A(3,2)
            A(mu + 2*M, nu + Nprime) = t1.transpose() * H_int2; 

            // A(4,1)
            A(mu + 3*M, nu) = t2.transpose() * H_int1; 
            
            // A(4,2)
            A(mu + 3*M, nu + Nprime) = t2.transpose() * H_int2; 
        }

        for (int nu = 0; nu < N2prime; ++nu) {
            sources_ext[2*nu]->computeFields(E_HD, H_HD, x_mu);

            Vector3cd E_ext1 = E_HD.row(0);
            Vector3cd H_ext1 = H_HD.row(0);

            sources_ext[2*nu + 1]->computeFields(E_HD, H_HD, x_mu);

            Vector3cd E_ext2 = E_HD.row(0);
            Vector3cd H_ext2 = H_HD.row(0);

            // Electric fields
            // A(1,3) 
            A(mu, nu + 2*Nprime) = -t1.transpose() * E_ext1;
            
            // A(1,4)
            A(mu, nu + 2*Nprime + N2prime) = -t1.transpose() * E_ext2;

            // A(2,3)
            A(mu + M, nu + 2*Nprime) = -t2.transpose() * E_ext1;
            
            // A(2,4)
            A(mu + M, nu + 2*Nprime + N2prime)  = -t2.transpose() * E_ext2;

            // Magnetic fields
            // A(3,3)
            A(mu + 2*M, nu + 2*Nprime) = -t1.transpose() * H_ext1;
            
            // A(3,4)
            A(mu + 2*M, nu + 2*Nprime + N2prime) = -t1.transpose() * H_ext2;

            // A(4,3)
            A(mu + 3*M, nu + 2*Nprime) = -t2.transpose() * H_ext1;
            
            // A(4,4)
            A(mu + 3*M, nu + 2*Nprime + N2prime) = -t2.transpose() * H_ext2;
        }
    }
}
