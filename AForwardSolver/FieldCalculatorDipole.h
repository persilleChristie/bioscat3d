#ifndef FIELDCALCULATORDIPOLE_H
#define FIELDCALCULATORDIPOLE_H

#include "FieldCalculator.h"
#include "UtilsDipole.h"
#include "Constants.h"
#include <Eigen/Dense>
#include <complex>

class FieldCalculatorDipole : public FieldCalculator {
public:
    FieldCalculatorDipole(const Dipole& dipole, const Constants& constants);

    virtual void computeFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints
    ) const override;

private:
    Dipole dipole_;
    Constants constants_;
};

#endif // FIELDCALCULATORDIPOLE_H
