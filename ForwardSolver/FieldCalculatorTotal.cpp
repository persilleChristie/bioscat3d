#include "FieldCalculatorTotal.h"

FieldCalculatorTotal::FieldCalculatorTotal(
    const std::vector<std::complex<double>>& amplitudes,
    const std::vector<std::shared_ptr<FieldCalculator>>& calculators
)
    : amplitudes_(amplitudes), calculators_(calculators)
{
    if (amplitudes_.size() != calculators_.size()) {
        throw std::invalid_argument("Amplitudes and calculators must have the same size.");
    }
}

void FieldCalculatorTotal::computeFields(
    Eigen::MatrixX3cd& outE,
    Eigen::MatrixX3cd& outH,
    const Eigen::MatrixX3d& evalPoints
) const {
    int N = evalPoints.rows();
    outE = Eigen::MatrixX3cd::Zero(N, 3);
    outH = Eigen::MatrixX3cd::Zero(N, 3);

    Eigen::MatrixX3cd Ei, Hi;

    for (size_t i = 0; i < calculators_.size(); ++i) {
        calculators_[i]->computeFields(Ei, Hi, evalPoints);
        outE += amplitudes_[i] * Ei;
        outH += amplitudes_[i] * Hi;
    }
}
