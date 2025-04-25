#ifndef FIELDCALCULATORTOTAL_H
#define FIELDCALCULATORTOTAL_H

#include "FieldCalculator.h"
#include "FieldCalculatorDipole.h"
#include "FieldCalculatorUPW.h"
#include "Constants.h"
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "Surface.h"
#include <fstream>
#include <iostream>

class FieldCalculatorTotal : public FieldCalculator {
public:
    FieldCalculatorTotal(
        std::string testpoint_file,
        std::string aux_point_file,
        // const Eigen::VectorXcd & amplitudes,
        // const std::vector<std::shared_ptr<FieldCalculator>>& dipoles,
        const std::shared_ptr<FieldCalculator>& UPW,
        Constants constants
    );

    void computeFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints
    ) const override;

    double computePower(
        const Surface& surface
    );

    Eigen::VectorXcd getAmplitudes() {return amplitudes_;}

private:
    Eigen::VectorXcd amplitudes_;
    std::vector<std::shared_ptr<FieldCalculator>> dipoles_;
    std::shared_ptr<FieldCalculator> UPW_;
    Constants constants_;

    void constructor(std::string testpoint_file,
                     std::string aux_point_file);
};

#endif // FIELDCALCULATORTOTAL_H
