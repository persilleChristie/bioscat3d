// #pragma once
#include <Eigen/Dense>
#include <cmath>
#include <random>

namespace GaussianProcess{
//
// --- RBF Kernel ---
// k(x, x') = σ_f² * exp(-||x - x'||² / (2ℓ²))
// 
inline double rbf_kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
                         double lengthScale, double sigmaF) {
    double r2 = (x1 - x2).squaredNorm();
    return sigmaF * sigmaF * std::exp(-r2 / (2.0 * lengthScale * lengthScale));
}

//
// --- Matern 3/2 Kernel ---
// k(x, x') = σ_f² * (1 + sqrt(3) r / ℓ) * exp(-sqrt(3) r / ℓ)
//
inline double matern32_kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
                              double lengthScale, double sigmaF) {
    double r = (x1 - x2).norm();
    double s = std::sqrt(3.0) * r / lengthScale;
    return sigmaF * sigmaF * (1.0 + s) * std::exp(-s);
}

//
// --- Matern 5/2 Kernel ---
// k(x, x') = σ_f² * (1 + sqrt(5) r / ℓ + 5r²/(3ℓ²)) * exp(-sqrt(5) r / ℓ)
//
inline double matern52_kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
                              double lengthScale, double sigmaF) {
    double r = (x1 - x2).norm();
    double s = std::sqrt(5.0) * r / lengthScale;
    return sigmaF * sigmaF * (1.0 + s + (5.0 / 3.0) * r * r / (lengthScale * lengthScale)) * std::exp(-s);
}

//
// --- Zero Mean Function ---
//
inline Eigen::VectorXd zero_mean(const Eigen::MatrixXd& X) {
    return Eigen::VectorXd::Zero(X.rows());

}


inline Eigen::VectorXd sample_gp_2d_fast(const Eigen::MatrixXd& X,
                                  double length_scale,
                                  double sigma_f,
                                  std::mt19937& gen) {
    const int N = X.rows();
    const double inv_2l2 = 0.5 / (length_scale*length_scale);
    const double sigmaf2     = sigma_f * sigma_f;
    const double jitter  = 1e-6;

    // 1) Distance matrix via BLAS
    Eigen::VectorXd sqnorm = X.rowwise().squaredNorm();
    Eigen::MatrixXd G      = X * X.transpose();
    Eigen::MatrixXd dist2  = sqnorm.replicate(1, N)
                           + sqnorm.transpose().replicate(N, 1)
                           - 2.0 * G;

    // 2) Covariance
    Eigen::MatrixXd K = (-inv_2l2 * dist2.array())
                        .exp()
                        .matrix() * sigmaf2;
    K.diagonal().array() += jitter;

    // 3) Cholesky on self-adjoint view
    Eigen::LLT<Eigen::MatrixXd> llt(K.selfadjointView<Eigen::Lower>());
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("Cholesky failed");

    // 4) Sample z ~ N(0,I)
    std::normal_distribution<> N01(0.0,1.0);
    Eigen::VectorXd z(N);
    for (int i = 0; i < N; ++i) z[i] = N01(gen);

    // 5) Return L * z
    return llt.matrixL() * z;
}
}