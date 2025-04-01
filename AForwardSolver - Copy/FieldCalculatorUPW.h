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

public:
    FieldCalculatorUPW(const Eigen::Vector3d& k_in, const Eigen::Vector3d& E0_in)
        : k(k_in), E0(E0_in.cast<std::complex<double>>()){}

    void computeFields(
        Eigen::MatrixXd& outE,
        Eigen::MatrixXd& outH,
        const Eigen::MatrixXd& evalPoints
    ) const override;

    virtual void computeFields(
        Eigen::MatrixXd& outE,
        Eigen::MatrixXd& outH,
        const Eigen::MatrixXd& evalPoints
    ) const = 0;

    Eigen::Vector3cd getEField(const Eigen::Vector3d& x) const override;
    Eigen::Vector3cd getHField(const Eigen::Vector3d& x) const override;
};

#endif // FIELDCALCULATORUPW_H
