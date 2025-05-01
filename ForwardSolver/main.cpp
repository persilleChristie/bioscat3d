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
    const char* jsonPath = "../MeepTests/SetupTests/SurfaceData/surfaceParams.json";

    MASSystem masSystem(jsonPath, "Bump", constants);

    FieldCalculatorTotal field(masSystem, constants);

    // Top
    double size1 = 2, size2 = 2;
    Eigen::Vector3d Cornerpoint (-1, -1, 10);
    Eigen::Vector3d basis1 (1,0,0), basis2 (0,1,0);
    SurfacePlane top(Cornerpoint, basis1, basis2, size1, size2, 20);
    Eigen::VectorXd powers = field.computePower(top);

    Export::saveRealVectorCSV("FilesCSV/powers_polarization.csv", powers);

    ///// PARAMETERS /////

    // // Incident field
    // Vector3d k_inc(1.0, 1.0, -1.0);
    // k_inc = k_inc.normalized();
    // // double polarization = 0.2;
    // double E0 = 1.0;

    // // Simulation parameters
    // // double width = 2.0;
    // // int resol = 10; 
    // // double pml_thickness = 2;
    // // int resolution = resol;
    // // double dim = width + 2 * pml_thickness;

    // // double cell_x = dim;
    // // double cell_y = dim;
    // // double cell_z = dim;

    // double waveLength = 325e-3;
    // // double frequency = 1/waveLength;

    // constants.setWavelength(waveLength);
    
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


    
    // Eigen::VectorXcd amplitudes = y.segment(0,2*Nprime);
    // FieldCalculatorTotal field(amplitudes, sources_int, incident);

    // double size1 = dim, size2 = dim;
    
    // // Top
    // Eigen::Vector3d Cornerpoint (-cell_x, -cell_y, 0.5*cell_z - pml_thickness - 0.1);
    // Eigen::Vector3d basis1 (1,0,0), basis2 (0,1,0);
    // SurfacePlane top(Cornerpoint, basis1, basis2, size1, size2, resolution);


    // // Bottom
    // Cornerpoint << -cell_x, -cell_y, -0.5*cell_z + pml_thickness + 0.1;
    // SurfacePlane bottom(Cornerpoint, basis1, basis2, size1, size2, resolution);


    // // Left
    // Cornerpoint << -0.5*cell_x + pml_thickness + 0.1, -cell_y, -cell_z;
    // basis1 << 0,1,0;
    // basis2 << 0,0,1;
    // SurfacePlane left(Cornerpoint, basis1, basis2, size1, size2, resolution);

    
    // // Right
    // Cornerpoint << 0.5*cell_x - pml_thickness - 0.1, -cell_y, -cell_z;
    // SurfacePlane right(Cornerpoint, basis1, basis2, size1, size2, resolution);

    

    // // Back
    // Cornerpoint << -cell_x, -0.5*cell_y + pml_thickness + 0.1, -cell_z;
    // basis1 << 1,0,0;
    // basis2 << 0,0,1;
    // SurfacePlane back(Cornerpoint, basis1, basis2, size1, size2, resolution);

    // // Front
    // Cornerpoint << -cell_x, 0.5*cell_y - pml_thickness - 0.1, -cell_z;
    // SurfacePlane front(Cornerpoint, basis1, basis2, size1, size2, resolution);

    // Top
    // double size1 = 2, size2 = 2;
    // Eigen::Vector3d Cornerpoint (-1, -1, 10);
    // Eigen::Vector3d basis1 (1,0,0), basis2 (0,1,0);
    // SurfacePlane top(Cornerpoint, basis1, basis2, size1, size2, 20);

    // int pol_nr = 100;
    // Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(pol_nr, 0, 0.5 * constants.pi);
    // // std::cout << "Polarization: " << betas << std::endl;
    // double beta;
    // Eigen::VectorXd powers(pol_nr);
    
    // for (int i = 0; i < pol_nr; ++i){
    //     beta = betas(i);
    //     // Calculate total field and power
    //     std::shared_ptr<FieldCalculatorUPW> incident = std::make_shared<FieldCalculatorUPW>(k_inc, E0, beta, constants);
        
    //     FieldCalculatorTotal field("../MeepTests/SurfaceData/test_points.csv", 
    //                                 "../MeepTests/SurfaceData/aux_points.csv", 
    //                                 incident,
    //                                 constants);

    //     Eigen::MatrixX3d points = top.getPoints();

    //     int N = points.rows();

    //     Eigen::MatrixX3cd outE, outH;
    //     outE = Eigen::MatrixX3cd::Zero(N,3);
    //     outH = Eigen::MatrixX3cd::Zero(N,3);
        
    //     field.computeFields(outE, outH, points);

    //     Export::saveMatrixCSV("FilesCSV/scatteredFieldE_PN.csv", outE);
    //     Export::saveMatrixCSV("FilesCSV/scatteredFieldH_PN.csv", outH);
        
    //     double power_top = field.computePower(top);

    //     powers(i) = power_top;

    //     // double power_bottom = field.computePower(bottom);
    //     // double power_left = field.computePower(left);
    //     // double power_right = field.computePower(right);
    //     // double power_front = field.computePower(front);
    //     // double power_back = field.computePower(back);

    //     // std::cout << "----------- Calculated power for beta = " << beta << ": -----------" << std::endl;
    //     // std::cout << "Top:    " << power_top << std::endl;
    //     // std::cout << "Bottom: " << power_bottom << std::endl;
    //     // std::cout << "Left:   " << power_left << std::endl;
    //     // std::cout << "Right:  " << power_right << std::endl;
    //     // std::cout << "Front:  " << power_front << std::endl;
    //     // std::cout << "Back:   " << power_back << std::endl;

    // }

    // Export::saveRealVectorCSV("FilesCSV/powers_polarization.csv", powers);

    return 0;
}
