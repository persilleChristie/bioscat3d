#ifndef FIELDCALCULATORTOTAL_H
#define FIELDCALCULATORTOTAL_H

#include "FieldCalculator.h"
#include "FieldCalculatorDipole.h"
#include "FieldCalculatorUPW.h"
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "Surface.h"

class FieldCalculatorTotal : public FieldCalculator {
public:
    FieldCalculatorTotal(
        const Eigen::VectorXcd & amplitudes,
        const std::vector<std::shared_ptr<FieldCalculatorDipole>>& dipoles,
        const std::shared_ptr<FieldCalculatorUPW>& UPW
    );

    void computeFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints
    ) const override;

    double computePower(
        const Surface& surface
    );

private:
    Eigen::Vector3cd amplitudes_;
    std::vector<std::shared_ptr<FieldCalculatorDipole>> dipoles_;
    std::shared_ptr<FieldCalculatorUPW> UPW_;
};

#endif // FIELDCALCULATORTOTAL_H
