#ifndef FIELDCALCULATORUPW_H
#define FIELDCALCULATORUPW_H

#include "FieldCalculator.h"
#include "Constants.h"
#include <Eigen/Dense>
#include <complex>

class FieldCalculatorUPW : public FieldCalculator {
private:
    Eigen::Vector3d k;
    double E0;
    double polarization;
    Constants constants;

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
        const Eigen::MatrixX3d& evalPoints
    ) const override;

};

#endif // FIELDCALCULATORUPW_H
