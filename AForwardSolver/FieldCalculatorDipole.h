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
};

#endif // FIELDCALCULATORDIPOLE_H
