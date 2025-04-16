#include "SystemAssembler.h"
#include "Surface.h"
#include "UtilsSolvers.h"
#include "FieldCalculator.h"
#include "FieldCalculatorDipole.h"
#include "FieldCalculatorUPW.h"
#include "Constants.h"
#include <iostream>
#include "UtilsFresnel.h"

using namespace Eigen;

void SystemAssembler::assembleSystem(
    Eigen::MatrixXcd& A,
    Eigen::VectorXcd& b,
    const Surface& surface,
    const std::vector<std::shared_ptr<FieldCalculator>>& sources_int,
    // const std::vector<std::shared_ptr<FieldCalculator>>& sources_mirr,
    const std::vector<std::shared_ptr<FieldCalculator>>& sources_ext,
    const std::shared_ptr<FieldCalculator>& incident
    // const std::complex<double> Gamma_r
) {
    const auto& points = surface.getPoints();
    const auto& tau1 = surface.getTau1();
    const auto& tau2 = surface.getTau2();

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

        b(mu)       = -E_inc_new.row(mu).dot(t1);
        b(mu + M)   = -E_inc_new.row(mu).dot(t2);
        b(mu + 2*M) = -H_inc_new.row(mu).dot(t1);
        b(mu + 3*M) = -H_inc_new.row(mu).dot(t2);
        

        for (int nu = 0; nu < Nprime; ++nu) {
            // sources_ext[nu]->computeFields(E_HD, H_HD, x_mu);
            sources_ext[2*nu]->computeFields(E_HD, H_HD, x_mu);

            Vector3cd E_int1 = E_HD.row(0);
            Vector3cd H_int1 = H_HD.row(0);

            // sources_ext[nu + N2prime]->computeFields(E_HD, H_HD, x_mu);
            sources_ext[2*nu + 1]->computeFields(E_HD, H_HD, x_mu);

            Vector3cd E_int2 = E_HD.row(0);
            Vector3cd H_int2 = H_HD.row(0);
            
            // sources_mirr[nu]->computeFields(E_HD, H_HD, x_mu);

            // Vector3cd E_mirr1 = E_HD.row(0);
            // Vector3cd H_mirr1 = H_HD.row(0);

            // sources_mirr[nu + Nprime]->computeFields(E_HD, H_HD, x_mu);

            // Vector3cd E_mirr2 = E_HD.row(0);
            // Vector3cd H_mirr2 = H_HD.row(0);


            // Electric fields
            // A(1,1) 
            A(mu, nu) = E_int1.dot(t1); // + Gamma_r * E_mirr1.dot(t1);
            
            // A(1,2)
            A(mu, nu + Nprime) = E_int2.dot(t1); // + Gamma_r * E_mirr2.dot(t1);

            // A(2,1)
            A(mu + M, nu) = E_int1.dot(t2); // + Gamma_r * E_mirr1.dot(t2);
            
            // A(2,2)
            A(mu + M, nu + Nprime) = E_int2.dot(t2); // + Gamma_r * E_mirr2.dot(t2);

            // Magnetic fields
            // A(3,1)
            A(mu + 2*M, nu) = H_int1.dot(t1); // + Gamma_r * H_mirr1.dot(t1);
            
            // A(3,2)
            A(mu + 2*M, nu + Nprime) = H_int2.dot(t1); // + Gamma_r * H_mirr2.dot(t1);

            // A(4,1)
            A(mu + 3*M, nu) = H_int1.dot(t2); // + Gamma_r * H_mirr1.dot(t2);
            
            // A(4,2)
            A(mu + 3*M, nu + Nprime) = H_int2.dot(t2); // + Gamma_r * H_mirr2.dot(t2);
        }

        for (int nu = 0; nu < N2prime; ++nu) {
            // sources_ext[nu]->computeFields(E_HD, H_HD, x_mu);
            sources_ext[2*nu]->computeFields(E_HD, H_HD, x_mu);

            Vector3cd E_ext1 = E_HD.row(0);
            Vector3cd H_ext1 = H_HD.row(0);

            // sources_ext[nu + N2prime]->computeFields(E_HD, H_HD, x_mu);
            sources_ext[2*nu + 1]->computeFields(E_HD, H_HD, x_mu);

            Vector3cd E_ext2 = E_HD.row(0);
            Vector3cd H_ext2 = H_HD.row(0);

            // std::cout << "Eext1" << E_ext1 << std::endl;
            // std::cout << "Eext2" << E_ext2 << std::endl;
            // std::cout << "Hext1" << H_ext1 << std::endl;
            // std::cout << "Hext2" << H_ext2 << std::endl;
        
            // std::cout << "E/H 1: " << E_ext1.norm()/H_ext1.norm() << std::endl;
            // std::cout << "E/H 2: " << E_ext2.norm()/H_ext2.norm() << std::endl;

            // Electric fields
            // A(1,3) 
            A(mu, nu + 2*Nprime) = E_ext1.dot(t1);
            
            // A(1,4)
            A(mu, nu + 2*Nprime + N2prime) = E_ext2.dot(t1);

            // A(2,3)
            A(mu + M, nu + 2*Nprime) = E_ext1.dot(t2);
            
            // A(2,4)
            A(mu + M, nu + 2*Nprime + N2prime)  = E_ext2.dot(t2);

            // Magnetic fields
            // A(3,3)
            A(mu + 2*M, nu + 2*Nprime) = H_ext1.dot(t1);
            
            // A(3,4)
            A(mu + 2*M, nu + 2*Nprime + N2prime) = H_ext2.dot(t1);

            // A(4,3)
            A(mu + 3*M, nu + 2*Nprime) = H_ext1.dot(t2);
            
            // A(4,4)
            A(mu + 3*M, nu + 2*Nprime + N2prime) = H_ext2.dot(t2);
        }
    }
}
