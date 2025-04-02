#pragma once
#include <Eigen/Dense>
#include <functional>

using KernelFunc = std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>;
using MeanFunc = std::function<Eigen::VectorXd(const Eigen::MatrixXd&)>;

class GaussianProcessModel {
public:
    GaussianProcessModel(KernelFunc kernel,
                         MeanFunc mean = [](const Eigen::MatrixXd& X) { return Eigen::VectorXd::Zero(X.rows()); },
                         double sigmaN = 1e-3);

    void setTrainingData(const Eigen::MatrixXd& X_train, const Eigen::VectorXd& y_train);
    Eigen::VectorXd evaluatePosteriorMean(const Eigen::MatrixXd& X_test) const;
    Eigen::MatrixXd evaluatePosteriorCovariance(const Eigen::MatrixXd& X_test) const;

private:
    KernelFunc kernel_;
    MeanFunc mean_;
    double sigmaN_;

    Eigen::MatrixXd X_train_;
    Eigen::VectorXd y_train_;
    Eigen::MatrixXd K_;
    Eigen::LLT<Eigen::MatrixXd> chol_;
    Eigen::VectorXd alpha_;  // Cached solution: K^-1(y - mean)
};
