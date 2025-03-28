#include "LinearSystemSolver.h"
#include "SystemAssembler.h"
#include "SystemSolver.h"
#include "SurfaceSphere.h"
#include "FieldCalculatorDipole.h"
#include "FieldCalculatorUPW.h"
#include "Dipole.h"
#include "Constants.h"
#include <memory>

using namespace Eigen;

LinearSystemSolver::Result LinearSystemSolver::solveSystem(double radius,
                                                           const Eigen::Vector3d& center,
                                                           int resolution,
                                                           double k0,
                                                           double Gamma_r,
                                                           const Eigen::Vector3d& k_inc,
                                                           const Eigen::Vector3d& polarization) {
    

    Constants constants;
    constants.k0 = k0; // Override k0 in constants if used elsewhere

    SurfaceSphere sphere_mu(radius, center, resolution);
    SurfaceSphere sphere_nu_prime(0.8 * radius, center, resolution);
    SurfaceSphere sphere_nu_2prime(1.2 * radius, center, resolution);
    auto sphere_nu_prime_tilde_ptr = sphere_nu_prime.mirrored(Vector3d(0, 0, 1));
    auto* sphere_nu_prime_tilde = dynamic_cast<SurfaceSphere*>(sphere_nu_prime_tilde_ptr.get());

    std::vector<std::shared_ptr<FieldCalculator>> sources;

    auto addDipoles = [&](const SurfaceSphere& surface) {
        const MatrixXd& pts = surface.getPoints();
        const MatrixXd& t1 = surface.getTau1();
        const MatrixXd& t2 = surface.getTau2();
        for (int i = 0; i < pts.rows(); ++i) {
            sources.push_back(std::make_shared<FieldCalculatorDipole>(Dipole(pts.row(i), t1.row(i))));
            sources.push_back(std::make_shared<FieldCalculatorDipole>(Dipole(pts.row(i), t2.row(i))));
        }
    };

    addDipoles(sphere_nu_prime);
    addDipoles(*sphere_nu_prime_tilde);
    addDipoles(sphere_nu_2prime);

    auto incident = std::make_shared<FieldCalculatorUPW>(k_inc, polarization);

    MatrixXcd A;
    VectorXcd b;
    SystemAssembler::assemble(A, b, sphere_mu, sources, incident);

    Eigen::VectorXcd x = SystemSolver::solve(A, b);

    return { A, b, x };
}
