#ifndef FIELDCALCULATORDIPOLE_H
#define FIELDCALCULATORDIPOLE_H

#include "FieldCalculator.h"
#include "Dipole.h"
#include <Eigen/Dense>

class FieldCalculatorDipole : public FieldCalculator {
public:
    FieldCalculatorDipole(const Dipole& dipole);

    void computeFields(
        Eigen::MatrixXd& outE,
        Eigen::MatrixXd& outH,
        const Eigen::MatrixXd& evalPoints
    ) const override;

private:
    Dipole dip;

    static constexpr double mu0 = 4 * M_PI * 1e-7;
    static constexpr double epsilon0 = 8.854187817e-12;
    static constexpr double c0 = 1.0 / std::sqrt(mu0 * epsilon0);
};

#endif // FIELDCALCULATORDIPOLE_H
