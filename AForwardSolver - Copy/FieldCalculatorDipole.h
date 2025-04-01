#ifndef FIELDCALCULATORDIPOLE_H
#define FIELDCALCULATORDIPOLE_H

#include "FieldCalculator.h"
#include "UtilsDipole.h"
#include "Constants.h"
#include <Eigen/Dense>
#include <complex>

class FieldCalculatorDipole : public FieldCalculator {
public:
    FieldCalculatorDipole(const Dipole& dipole, double Iel = 1.0, const Constants& constants = Constants());

    virtual void computeFields(
        Eigen::MatrixXd& outE,
        Eigen::MatrixXd& outH,
        const Eigen::MatrixXd& evalPoints
    ) const override;

    std::pair<Eigen::Vector3cd, Eigen::Vector3cd> getFields(const Eigen::Vector3d& x) const;

private:
    Dipole dipole_;
    double Iel_;
    Constants constants_;

    std::pair<Eigen::Vector3cd, Eigen::Vector3cd> computeFieldAt(const Eigen::Vector3d& x) const;
};

#endif // FIELDCALCULATORDIPOLE_H
