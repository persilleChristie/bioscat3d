<<<<<<< HEAD
#ifndef FIELDCALCULATOR_H
#define FIELDCALCULATOR_H

#include <Eigen/Dense>

class FieldCalculator {
public:
    // Pointwise field access (needed by SystemAssembler)
    virtual Eigen::Vector3cd getEField(const Eigen::Vector3d& x) const = 0;
    virtual Eigen::Vector3cd getHField(const Eigen::Vector3d& x) const = 0;

    // Batched field computation
    virtual void computeFields(
        Eigen::MatrixXd& outE,
        Eigen::MatrixXd& outH,
        const Eigen::MatrixXd& evalPoints
    ) const = 0;

    // Tangential components
    virtual void computeTangentialFields(
        Eigen::MatrixXd& outEt,
        Eigen::MatrixXd& outHt,
        const Eigen::MatrixXd& evalPoints,
        const Eigen::MatrixXd& tau1,
        const Eigen::MatrixXd& tau2
    ) const {
        Eigen::MatrixXd E, H;
        computeFields(E, H, evalPoints);

        int N = evalPoints.rows();
        outEt.resize(N, 2);
        outHt.resize(N, 2);

        for (int i = 0; i < N; ++i) {
            outEt(i, 0) = E.row(i).dot(tau1.row(i));
            outEt(i, 1) = E.row(i).dot(tau2.row(i));
            outHt(i, 0) = H.row(i).dot(tau1.row(i));
            outHt(i, 1) = H.row(i).dot(tau2.row(i));
        }
    }

    virtual ~FieldCalculator() = default;
};

#endif // FIELDCALCULATOR_H
=======
#ifndef FIELDCALCULATOR_H
#define FIELDCALCULATOR_H

#include <Eigen/Dense>

class FieldCalculator {
public:
    // Compute full E and H fields at given evaluation points
    virtual void computeFields(
        Eigen::MatrixXd& outE,
        Eigen::MatrixXd& outH,
        const Eigen::MatrixXd& evalPoints
    ) const = 0;

    // Compute tangential components of E and H given tau1 and tau2 directions
    virtual void computeTangentialFields(
        Eigen::MatrixXd& outEt,
        Eigen::MatrixXd& outHt,
        const Eigen::MatrixXd& evalPoints,
        const Eigen::MatrixXd& tau1,
        const Eigen::MatrixXd& tau2
    ) const {
        Eigen::MatrixXd E, H;
        computeFields(E, H, evalPoints);

        int N = evalPoints.rows();
        outEt.resize(N, 2);
        outHt.resize(N, 2);

        for (int i = 0; i < N; ++i) {
            outEt(i, 0) = E.row(i).dot(tau1.row(i));
            outEt(i, 1) = E.row(i).dot(tau2.row(i));
            outHt(i, 0) = H.row(i).dot(tau1.row(i));
            outHt(i, 1) = H.row(i).dot(tau2.row(i));
        }
    }

    virtual ~FieldCalculator() = default;
};

#endif // FIELDCALCULATOR_H
>>>>>>> a5573bfeda0575951aeed785824f8fe3a44f17d7
