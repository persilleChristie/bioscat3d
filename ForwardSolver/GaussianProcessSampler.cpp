#include "GaussianProcessSampler.h"
#include "SurfaceFromGP.h"
#include <Eigen/Eigenvalues>
#include <random>

GaussianProcessSampler::GaussianProcessSampler(const GaussianProcessModel& gp_model)
    : gp_(gp_model), rng_(std::random_device{}()) {}

std::unique_ptr<Surface> GaussianProcessSampler::samplePosteriorSurface(const Eigen::MatrixXd& X_test) {
    const int m = X_test.rows();

    Eigen::VectorXd mu = gp_.evaluatePosteriorMean(X_test);
    Eigen::MatrixXd cov = gp_.evaluatePosteriorCovariance(X_test);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen decomposition failed for posterior covariance");

    Eigen::VectorXd z(m);
    std::normal_distribution<> dist(0.0, 1.0);
    for (int i = 0; i < m; ++i)
        z(i) = dist(rng_);

    Eigen::VectorXd sample = mu + solver.operatorSqrt() * z;

    Eigen::MatrixXd points(m, 3);
    points << X_test, sample;
    return std::make_unique<SurfaceFromGP>(points);
}

std::unique_ptr<Surface> GaussianProcessSampler::samplePriorSurface(const Eigen::MatrixXd& X_test) {
    const int m = X_test.rows();

    Eigen::MatrixXd cov(m, m);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            double val = gp_.evaluatePosteriorCovariance(X_test)(i, j); // inefficient, could cache!
            cov(i, j) = val;
            cov(j, i) = val;
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen decomposition failed for prior covariance");

    Eigen::VectorXd z(m);
    std::normal_distribution<> dist(0.0, 1.0);
    for (int i = 0; i < m; ++i)
        z(i) = dist(rng_);

    Eigen::VectorXd sample = solver.operatorSqrt() * z;

    Eigen::MatrixXd points(m, 3);
    points << X_test, sample;
    return std::make_unique<SurfaceFromGP>(points);
}