#include "LinearSystemSolver.h"
#include "SystemSolver.h"

LinearSystemSolver::Result LinearSystemSolver::solveSystem(
    const SphereSurface& sphere_mu,
    const SphereSurface& sphere_nu_prime,
    const SphereSurface& sphere_nu_prime_tilde,
    const SphereSurface& sphere_nu_2prime,
    const std::complex<double>& Gamma_r,
    const Eigen::Vector3d& k_inc,
    const Eigen::Vector3cd& E_inc,
    double k0,
    double Iel
) {
    using namespace Eigen;

    std::vector<FieldPtr> sources;

    // Build dipoles from tangent vectors
    auto addDipoles = [&](const SphereSurface& surface) {
        const MatrixXd& points = surface.getPoints();
        const MatrixXd& tau1 = surface.getTau1();
        const MatrixXd& tau2 = surface.getTau2();

        for (int i = 0; i < points.rows(); ++i) {
            sources.push_back(std::make_shared<FieldCalculatorDipole>(Dipole(points.row(i), tau1.row(i))));
            sources.push_back(std::make_shared<FieldCalculatorDipole>(Dipole(points.row(i), tau2.row(i))));
        }
    };

    addDipoles(sphere_nu_prime);
    addDipoles(sphere_nu_prime_tilde);
    addDipoles(sphere_nu_2prime);

    // Define incident field
    Vector3d polarization = E_inc.real();
    auto incidentField = std::make_shared<FieldCalculatorUPW>(k_inc, polarization);

    // Assemble A and b
    MatrixXd A;
    VectorXd b;
    SystemAssembler::assemble(A, b, sphere_mu, sources, incidentField);

    // Solve system
    VectorXd x = SystemSolver::solve(A, b);

    return { A, b, x };
}