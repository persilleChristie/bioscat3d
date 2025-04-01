#include "Surface.h"
#include "SurfaceSphere.h"
#include "UtilsExport.h"
#include "Constants.h"
#include "SystemAssembler.h"
#include "FieldCalculatorDipole.h"
#include "FieldCalculatorUPW.h"
#include "UtilsSolvers.h"
#include <iostream>

Constants constants;

int main() {
    using namespace Eigen;

    // Define parameters
    int num_points = 10;
    double radius = 1.0;
    Vector3d center(0.0, 0.0, 1.0);
    double k0 = 2 * constants.pi;
    double Gamma_r = 1.0;
    Vector3d k_inc(0.0, 0.0, -1.0);
    int resolution = 5;
    double polarization = 0.2;
    
    
    // create spheres
    SurfaceSphere sphere_mu(radius, center, resolution);
    SurfaceSphere sphere_nu_prime(0.8 * radius, center, resolution);
    SurfaceSphere sphere_nu_2prime(1.2 * radius, center, resolution);
    auto sphere_nu_prime_tilde_ptr = sphere_nu_prime.mirrored(Vector3d(0, 0, 1));
    auto* sphere_nu_prime_tilde = dynamic_cast<SurfaceSphere*>(sphere_nu_prime_tilde_ptr.get());



    // call System Assembler
    
    Eigen::MatrixX3cd A;
    Eigen::VectorXcd b;

    std::shared_ptr<FieldCalculatorUPW> incident = std::make_shared<FieldCalculatorUPW>(k_inc, polarization);

    std::vector<std::shared_ptr<FieldCalculator>> sources;

    auto addDipoles = [&](const SurfaceSphere& surface) {
        const MatrixXd& pts = surface.getPoints();
        const MatrixXd& t1 = surface.getTau1();
        const MatrixXd& t2 = surface.getTau2();
        for (int i = 0; i < pts.rows(); ++i) {
            sources.emplace_back(std::make_shared<FieldCalculatorDipole>(Dipole(pts.row(i), t1.row(i))));
            sources.emplace_back(std::make_shared<FieldCalculatorDipole>(Dipole(pts.row(i), t2.row(i))));
        }
    };

    addDipoles(sphere_nu_prime);
    addDipoles(*sphere_nu_prime_tilde);
    addDipoles(sphere_nu_2prime);

    SystemAssembler::assembleSystem(A, b, sphere_mu, sources, incident);
    
    // solve with UtilsSolver
    auto y = UtilsSolvers::solveQR(A, b);

    // Output sizes
    std::cout << "Linear system A has size: " << A.rows() << " x " << A.cols() << std::endl;
    std::cout << "Right-hand side b has size: " << b.size() << std::endl;
    std::cout << "Solution y has size: " << y.size() << std::endl;

    // Save to files
    Export::saveMatrixCSV("matrix_A_simple.csv", A);
    Export::saveVectorCSV("vector_b_simple.csv", b);
    Export::saveVectorCSV("solution_y_simple.csv", y);

    return 0;
}
