#ifndef FIELDCALCULATORTOTAL_H
#define FIELDCALCULATORTOTAL_H

#include "FieldCalculator.h"
#include "FieldCalculatorDipole.h"
#include "FieldCalculatorUPW.h"
#include "Constants.h"
#include "MASSystem.h"
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "Surface.h"
#include <fstream>
#include <iostream>


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
        const MASSystem masSystem,
        Constants constants
    );

    void computeFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints,
        int polarization_idx
    ) const override;

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> computePower(
        const Surface& surface
    );

    Eigen::MatrixXcd getAmplitudes() {return amplitudes_;}

private:
    Eigen::MatrixXcd amplitudes_;
    std::vector<std::shared_ptr<FieldCalculator>> dipoles_;
    Constants constants_;

    void constructor(const MASSystem masSystem);
};

#endif // FIELDCALCULATORTOTAL_H
