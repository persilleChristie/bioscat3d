#ifndef FIELDCALCULATORTOTAL_H
#define FIELDCALCULATORTOTAL_H

#include "FieldCalculator.h"
#include <vector>
#include <memory>

class FieldCalculatorTotal : public FieldCalculator {
public:
    FieldCalculatorTotal(
        const std::vector<std::complex<double>>& amplitudes,
        const std::vector<std::shared_ptr<FieldCalculator>>& calculators
    );

    void computeFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints
    ) const override;

private:
    std::vector<std::complex<double>> amplitudes_;
    std::vector<std::shared_ptr<FieldCalculator>> calculators_;
};

#endif // FIELDCALCULATORTOTAL_H
