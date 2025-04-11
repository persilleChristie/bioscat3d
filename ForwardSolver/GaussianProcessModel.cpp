#include "GaussianProcessModel.h"
#include <stdexcept>

GaussianProcessModel::GaussianProcessModel(KernelFunc kernel,
                                           MeanFunc mean,
                                           double sigmaN)
    : kernel_(std::move(kernel)), mean_(std::move(mean)), sigmaN_(sigmaN) {}

void GaussianProcessModel::setTrainingData(const Eigen::MatrixXd& X_train, const Eigen::VectorXd& y_train) {
    X_train_ = X_train;
    y_train_ = y_train;
    const int n = X_train.rows();

    // Compute training covariance matrix K(X, X)
    K_.resize(n, n);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            double val = kernel_(X_train.row(i), X_train.row(j));
            K_(i, j) = val;
            K_(j, i) = val; // symmetry
        }
    }
    K_.diagonal().array() += sigmaN_ * sigmaN_;

    // Cholesky decomposition of (K + σ_n^2 I)
    chol_ = K_.llt();
    if (chol_.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky decomposition failed in GaussianProcessModel");
    }

    // Precompute alpha = K^-1 (y - mean(X))
    Eigen::VectorXd y_centered = y_train_ - mean_(X_train_);
    alpha_ = chol_.solve(y_centered);
}

Eigen::VectorXd GaussianProcessModel::evaluatePosteriorMean(const Eigen::MatrixXd& X_test) const {
    const int m = X_test.rows();
    const int n = X_train_.rows();
    Eigen::MatrixXd K_star(m, n);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            K_star(i, j) = kernel_(X_test.row(i), X_train_.row(j));
        }
    }

    return mean_(X_test) + K_star * alpha_;
}

Eigen::MatrixXd GaussianProcessModel::evaluatePosteriorCovariance(const Eigen::MatrixXd& X_test) const {
    const int m = X_test.rows();
    const int n = X_train_.rows();
    Eigen::MatrixXd K_star(m, n);
    Eigen::MatrixXd K_starstar(m, m);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            K_star(i, j) = kernel_(X_test.row(i), X_train_.row(j));
        }
        for (int j = i; j < m; ++j) {
            double val = kernel_(X_test.row(i), X_test.row(j));
            K_starstar(i, j) = val;
            K_starstar(j, i) = val; // symmetry
        }
    }

    Eigen::MatrixXd v = chol_.solve(K_star.transpose());
    return K_starstar - K_star * v;
}
