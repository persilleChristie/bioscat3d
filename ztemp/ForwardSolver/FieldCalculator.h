#ifndef FIELDCALCULATOR_H
#define FIELDCALCULATOR_H

#include <Eigen/Dense>

class FieldCalculator {
public:

    // Batched field computation
    virtual void computeFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        const Eigen::MatrixX3d& evalPoints, 
        int polarization_idx = 0 // Only used in total fields
    ) const = 0;

    // Tangential components
    virtual void computeTangentialFields(
        Eigen::MatrixX3cd& outE,
        Eigen::MatrixX3cd& outH,
        Eigen::MatrixX3cd& outEt1,
        Eigen::MatrixX3cd& outEt2,
        Eigen::MatrixX3cd& outHt1,
        Eigen::MatrixX3cd& outHt2,
        const Eigen::MatrixX3d& tau1,
        const Eigen::MatrixX3d& tau2
    ) const {

        int N = outE.rows();

        for (int i = 0; i < N; ++i) {
            outEt1.row(i) = outE.row(i).dot(tau1.row(i))*tau1.row(i);
            outEt2.row(i) = outE.row(i).dot(tau2.row(i))*tau2.row(i);
            outHt1.row(i) = outH.row(i).dot(tau1.row(i))*tau1.row(i);
            outHt2.row(i) = outH.row(i).dot(tau2.row(i))*tau2.row(i);
        }
    }

    virtual ~FieldCalculator() = default;
};

#endif // FIELDCALCULATOR_H
