<<<<<<< HEAD
#ifndef FIELDCALCULATORDIPOLE_H
#define FIELDCALCULATORDIPOLE_H

#include "FieldCalculator.h"
#include "Dipole.h"
#include "Constants.h"
#include <Eigen/Dense>
#include <complex>

class FieldCalculatorDipole : public FieldCalculator {
public:
    FieldCalculatorDipole(const Dipole& dipole, double Iel = 1.0, const Constants& constants = Constants());

    void computeFields(
        Eigen::MatrixXd& outE_real,
        Eigen::MatrixXd& outH_real,
        const Eigen::MatrixXd& evalPoints
    ) const override;

    Eigen::Vector3cd getEField(const Eigen::Vector3d& x) const override;
    Eigen::Vector3cd getHField(const Eigen::Vector3d& x) const override;

private:
    Dipole dipole_;
    double Iel_;
    Constants constants_;

    std::pair<Eigen::Vector3cd, Eigen::Vector3cd> computeFieldAt(const Eigen::Vector3d& x) const;
};

#endif // FIELDCALCULATORDIPOLE_H
=======
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
>>>>>>> a5573bfeda0575951aeed785824f8fe3a44f17d7
