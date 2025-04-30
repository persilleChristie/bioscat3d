#ifndef FIELDCALCULATORUPW_H
#define FIELDCALCULATORUPW_H

#include "FieldCalculator.h"
#include "Constants.h"
#include <Eigen/Dense>
#include <complex>

class FieldCalculatorUPW : public FieldCalculator {
private:
    Eigen::Vector3d k_;
    double E0_;
    double polarization_;
    Constants constants_;

public:
    FieldCalculatorUPW(const Eigen::Vector3d& k_in, const double E0_in, const double polarization_in, 
                        const Constants& constants);


    void computeReflectedFields(
        Eigen::MatrixX3cd& refE,
        Eigen::MatrixX3cd& refH,
        const Eigen::MatrixX3d& evalPoints
    ) const;
    
    
    virtual void computeFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints,
        int polarization_idx = 0 // Only used in total fields
    ) const override;

};

#endif // FIELDCALCULATORUPW_H
