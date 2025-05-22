#ifndef FIELDCALCULATORTOTAL_H
#define FIELDCALCULATORTOTAL_H

#include "FieldCalculator.h"
#include "../Utils/Constants.h"
#include "MASSystem.h"
#include <Eigen/Dense>
#include "Surface.h"

/**
    * Class FieldCalculatorTotal, storing the scattered field amplitudes, the interior dipoles. Has methods to 
    * calculate the total power for a number of polarizations
    *  
    * @param jsonPath char with path to json file including the parameters needed for each surface:
    *       - Bump needs 'resolution', 'halfWidth_x', 'halfWidth_y', 'halfWidth_z', 'bumpData' (hereunder 
    *                   'x0', 'y0', 'height' and 'sigma'), 'k' (incidence vector) and 'betas' (list of polarizations)
    *       - GP needs ...
    * @param surfaceType string indicating type of surface - allowed types are 'Bump' and 'GP'
    * @param constants updated Constants file
*/
class FieldCalculatorTotal : public FieldCalculator {
public:
    FieldCalculatorTotal(
        const MASSystem masSystem
    );

    void computeFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints,
        int polarization_idx = 0
    ) const override;

    Eigen::VectorXd computePower(
        const Surface& surface
    );

    Eigen::MatrixXcd getAmplitudes() {return amplitudes_;}

    Eigen::VectorXd computeTangentialError();

private:
    Eigen::MatrixXcd amplitudes_;
    Eigen::MatrixXcd ampltudes_ext_;
    std::vector<std::shared_ptr<FieldCalculator>> dipoles_;
    std::vector<std::shared_ptr<FieldCalculator>> UPW_;
    MASSystem mas_;

    void constructor();
};

#endif // FIELDCALCULATORTOTAL_H
