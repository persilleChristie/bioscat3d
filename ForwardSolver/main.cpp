#include "Surface.h"
#include "SurfaceSphere.h"
#include "UtilsExport.h"
#include "Constants.h"
#include "SystemAssembler.h"
#include "FieldCalculatorDipole.h"
#include "FieldCalculatorUPW.h"
#include "UtilsSolvers.h"
#include <iostream>
#include "FieldCalculatorTotal.h"
#include "SurfacePlane.h"

Constants constants;

int main() {
    using namespace Eigen;

    ///// PARAMETERS /////

    // Incident field
    Vector3d k_inc(0.0, 0.0, -1.0);
    double polarization = 0.2;
    double E0 = 1.0;

    // Cell definitions, Meep


    // Other constants
    double dim = 2.0;
    double ;
    int resolution = 10;
    constants.setWavelength(325e-3);
    
    // create spheres
    // SurfaceSphere sphere_mu(radius, center, resolution);
    // SurfaceSphere sphere_nu_prime(0.8 * radius, center, resolution);
    // SurfaceSphere sphere_nu_2prime(1.2 * radius, center, resolution);
    
    // auto sphere_nu_prime_tilde_ptr = sphere_nu_prime.mirrored(Vector3d(0, 0, 1));
    // auto* sphere_nu_prime_tilde = dynamic_cast<SurfaceSphere*>(sphere_nu_prime_tilde_ptr.get());

    
    // call System Assembler
    // int M = sphere_mu.getPoints().rows();
    // int Nprime = sphere_nu_prime.getPoints().rows();
    // int N = Nprime + sphere_nu_2prime.getPoints().rows();
    
    // Eigen::MatrixXcd A(4 * M, 2*N);
    // Eigen::VectorXcd b(4 *M);

    // std::vector<std::shared_ptr<FieldCalculator>> sources_int;
    // // std::vector<std::shared_ptr<FieldCalculator>> sources_mirr;
    // std::vector<std::shared_ptr<FieldCalculator>> sources_ext;

    // auto addDipoles = [&](const SurfaceSphere& surface, std::vector<std::shared_ptr<FieldCalculator>>& sources) {
    //     const MatrixXd& pts = surface.getPoints();
    //     const MatrixXd& t1 = surface.getTau1();
    //     const MatrixXd& t2 = surface.getTau2();
    //     for (int i = 0; i < pts.rows(); ++i) {
    //         sources.emplace_back(std::make_shared<FieldCalculatorDipole>(Dipole(pts.row(i), t1.row(i)), constants));
    //         sources.emplace_back(std::make_shared<FieldCalculatorDipole>(Dipole(pts.row(i), t2.row(i)), constants));
    //     }
    // };

    // addDipoles(sphere_nu_prime, sources_int);
    // // addDipoles(*sphere_nu_prime_tilde, sources_mirr);
    // addDipoles(sphere_nu_2prime, sources_ext);


    // const auto& points = sphere_mu.getPoints();
    // const auto& tau1 = sphere_mu.getTau1();
    // const auto& tau2 = sphere_mu.getTau2();

    // SystemAssembler::assembleSystem(A, b, points, tau1, tau2, sources_int, sources_ext, incident);
    
    // // solve with UtilsSolver
    // auto y = UtilsSolvers::solveQR(A, b);
    
    // // Output sizes
    // std::cout << "Linear system A has size: " << A.rows() << " x " << A.cols() << std::endl;
    // std::cout << "Right-hand side b has size: " << b.size() << std::endl;
    // std::cout << "Solution y has size: " << y.size() << std::endl;

    // // Save to files
    // Export::saveMatrixCSV("matrix_A_simple.csv", A);
    // Export::saveVectorCSV("vector_b_simple.csv", b);
    // Export::saveVectorCSV("solution_y_simple.csv", y);


    // Calculate total field and power
    std::shared_ptr<FieldCalculatorUPW> incident = std::make_shared<FieldCalculatorUPW>(k_inc, E0, polarization, constants);
    
    FieldCalculatorTotal field("../MeepTests/SurfaceData/test_points.csv", 
                                "../MeepTests/SurfaceData/aux_points.csv", 
                                incident,
                                constants);
    
    // Eigen::VectorXcd amplitudes = y.segment(0,2*Nprime);
    // FieldCalculatorTotal field(amplitudes, sources_int, incident);

    Eigen::Vector3d Cornerpoint(-5.0,-5.0,10.0), basis1(1.0,1.0,0.0), basis2(-1.0,1.0,0.0);
    double size1 = 1.0, size2 = 1.0;
    
    SurfacePlane testPlane(Cornerpoint, basis1, basis2, size1, size2, resolution);

    double power = field.computePower(testPlane);

    std::cout << "Calculated power: " << power << std::endl;

    return 0;
}
