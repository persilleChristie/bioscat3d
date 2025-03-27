#include "SystemAssembler.h"

void SystemAssembler::assemble(
    Eigen::MatrixXcd& A,
    Eigen::VectorXcd& b,
    const Surface& surface,
    const std::vector<std::shared_ptr<FieldCalculator>>& sources,
    const std::shared_ptr<FieldCalculator>& incidentField
) {
    const Eigen::MatrixXd& points = surface.getPoints();
    const Eigen::MatrixXd& tau1 = surface.getTau1();
    const Eigen::MatrixXd& tau2 = surface.getTau2();

    int N = points.rows();
    int M = static_cast<int>(sources.size());

    A.resize(2 * N, M);
    b.resize(2 * N);

    // Fill matrix A from all sources
    for (int m = 0; m < M; ++m) {
        Eigen::MatrixXd Et, Ht;
        sources[m]->computeTangentialFields(Et, Ht, points, tau1, tau2);

        for (int i = 0; i < N; ++i) {
            A(2 * i,     m) = Et(i, 0); // tau1 component
            A(2 * i + 1, m) = Et(i, 1); // tau2 component
        }
    }

    // Compute RHS vector b from incident field
    Eigen::MatrixXd Et_inc, Ht_inc;
    incidentField->computeTangentialFields(Et_inc, Ht_inc, points, tau1, tau2);

    for (int i = 0; i < N; ++i) {
        b(2 * i)     = -Et_inc(i, 0); // tau1 component
        b(2 * i + 1) = -Et_inc(i, 1); // tau2 component
    }
}
