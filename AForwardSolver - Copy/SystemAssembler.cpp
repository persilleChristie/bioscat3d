#include "SystemAssembler.h"
#include "UtilsSolvers.h"
#include "FieldCalculator.h"
#include "FieldCalculatorDipole.h"
#include "FieldCalculatorUPW.h"
#include <iostream>

using namespace Eigen;

void SystemAssembler::assembleSystem(
    Eigen::MatrixX3cd& A,
    Eigen::VectorXcd& b,
    const Surface& surface,
    const std::vector<std::shared_ptr<FieldCalculator>>& sources,
    const std::shared_ptr<FieldCalculator>& incident
) {
    const auto& points = surface.getPoints();
    const auto& tau1 = surface.getTau1();
    const auto& tau2 = surface.getTau2();

    int M = points.rows();
    int N = sources.size();
    A.resize(4 * M, N);
    b.resize(4 * M);

    for (int mu = 0; mu < M; ++mu) {
        Vector3d x_mu = points.row(mu);
        Vector3d t1 = tau1.row(mu);
        Vector3d t2 = tau2.row(mu);

        Vector3cd E_inc = incident->getEField(x_mu);
        Vector3cd H_inc = incident->getHField(x_mu);

        b(mu)       = -E_inc.dot(t1);
        b(mu + M)   = -E_inc.dot(t2);
        b(mu + 2*M) = -H_inc.dot(t1);
        b(mu + 3*M) = -H_inc.dot(t2);

        for (int nu = 0; nu < N; ++nu) {
            Vector3cd E = sources[nu]->getEField(x_mu);
            Vector3cd H = sources[nu]->getHField(x_mu);

            A(mu, nu)       = E.dot(t1);
            A(mu + M, nu)   = E.dot(t2);
            A(mu + 2*M, nu) = H.dot(t1);
            A(mu + 3*M, nu) = H.dot(t2);
        }
    }
}
