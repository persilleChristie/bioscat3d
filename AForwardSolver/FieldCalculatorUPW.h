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
