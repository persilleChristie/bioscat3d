#ifndef FIELDCALCULATORDIPOLE_H
#define FIELDCALCULATORDIPOLE_H

#include "FieldCalculator.h"
#include "../Utils/UtilsDipole.h"
#include "../Utils/Constants.h"
#include <Eigen/Dense>

class FieldCalculatorDipole : public FieldCalculator {
public:
    FieldCalculatorDipole(const Dipole& dipole, const bool interior);

    virtual void computeFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints,
        int polarization_idx = 0 // Only used in total fields
    ) const override;

    const Eigen::Vector3d& getDirection() const { return dipole_.getDirection(); }

private:
    Dipole dipole_;
    bool interiorBool_;
};

#endif // FIELDCALCULATORDIPOLE_H
