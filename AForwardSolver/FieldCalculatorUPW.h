<<<<<<< HEAD
#ifndef FIELDCALCULATORUPW_H
#define FIELDCALCULATORUPW_H

#include "FieldCalculator.h"
#include "Constants.h"
#include <Eigen/Dense>
#include <complex>

class FieldCalculatorUPW : public FieldCalculator {
private:
    Eigen::Vector3d k;
    Eigen::Vector3cd E0;
    Constants constants;

public:
    FieldCalculatorUPW(const Eigen::Vector3d& k_in, const Eigen::Vector3d& E0_in)
        : k(k_in), E0(E0_in.cast<std::complex<double>>()), constants() {}

    void computeFields(
        Eigen::MatrixXd& outE,
        Eigen::MatrixXd& outH,
        const Eigen::MatrixXd& evalPoints
    ) const override;

    Eigen::Vector3cd getEField(const Eigen::Vector3d& x) const override;
    Eigen::Vector3cd getHField(const Eigen::Vector3d& x) const override;
};

#endif // FIELDCALCULATORUPW_H
=======
#ifndef UPWFIELDCALCULATOR_H
#define UPWFIELDCALCULATOR_H

#include "FieldCalculator.h"
#include <Eigen/Dense>

class FieldCalculatorUPW : public FieldCalculator {
public:
    FieldCalculatorUPW(const Eigen::Vector3d& waveVector,
                       const Eigen::Vector3d& polarization);

    void computeFields(
        Eigen::MatrixXd& outE,
        Eigen::MatrixXd& outH,
        const Eigen::MatrixXd& evalPoints
    ) const override;

private:
    Eigen::Vector3d k;   // wave vector
    Eigen::Vector3d E0;  // polarization vector
};

#endif // UPWFIELDCALCULATOR_H
>>>>>>> a5573bfeda0575951aeed785824f8fe3a44f17d7
