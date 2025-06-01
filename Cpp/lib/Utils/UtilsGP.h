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


inline Eigen::MatrixXd sample_gp_on_grid_rbf_fast(const Eigen::MatrixXd& X1,
                                                  const Eigen::MatrixXd& X2,
                                                  double length_scale,
                                                  double sigma_f,
                                                  std::mt19937& gen) {
    const int M = X1.rows();
    const int N = X1.cols();
    const int total_points = M * N;

    // Flatten meshgrid into (total_points, 2)
    Eigen::MatrixXd X(total_points, 2);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            X(i * N + j, 0) = X1(i, j);
            X(i * N + j, 1) = X2(i, j);
        }

    // Compute pairwise squared distances
    Eigen::VectorXd sqnorm = X.rowwise().squaredNorm();
    Eigen::MatrixXd G = X * X.transpose();
    Eigen::MatrixXd dist2 = sqnorm.replicate(1, total_points)
                          + sqnorm.transpose().replicate(total_points, 1)
                          - 2.0 * G;
    dist2 = dist2.cwiseMax(0.0); // Fix potential small negatives

    // Apply RBF kernel
    const double inv_2l2 = 0.5 / (length_scale * length_scale);
    const double sigmaf2 = sigma_f * sigma_f;
    Eigen::MatrixXd K = (-inv_2l2 * dist2.array()).exp().matrix() * sigmaf2;

    // Add jitter for numerical stability
    K.diagonal().array() += 1e-6;

    // Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> llt(K.selfadjointView<Eigen::Lower>());
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("Cholesky decomposition failed");

    // Sample standard normal vector
    std::normal_distribution<> N01(0.0, 1.0);
    Eigen::VectorXd z(total_points);
    for (int i = 0; i < total_points; ++i)
        z[i] = N01(gen);

    // Sample GP and reshape
    Eigen::VectorXd f = llt.matrixL() * z;
    Eigen::MatrixXd F(M, N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            F(i, j) = f(i * N + j);

    return F;
}


enum class MaternType { Matern32, Matern52 };

inline Eigen::MatrixXd sample_gp_on_grid_matern_fast(const Eigen::MatrixXd& X1,
                                                     const Eigen::MatrixXd& X2,
                                                     double length_scale,
                                                     double sigma_f,
                                                     std::mt19937& gen,
                                                     MaternType kernel_type) {
    const int M = X1.rows();
    const int N = X1.cols();
    const int total_points = M * N;

    // Flatten meshgrid into (total_points, 2)
    Eigen::MatrixXd X(total_points, 2);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            X(i * N + j, 0) = X1(i, j);
            X(i * N + j, 1) = X2(i, j);
        }

    // Compute pairwise squared Euclidean distances efficiently
    Eigen::VectorXd sqnorm = X.rowwise().squaredNorm();
    Eigen::MatrixXd G = X * X.transpose();
    Eigen::MatrixXd dist2 = sqnorm.replicate(1, total_points)
                          + sqnorm.transpose().replicate(total_points, 1)
                          - 2.0 * G;
    dist2 = dist2.cwiseMax(0.0); // Avoid small negative values

    // Convert squared distances to Euclidean distances
    Eigen::MatrixXd r = dist2.array().sqrt().matrix();

    // Apply selected Matern kernel
    Eigen::MatrixXd K(total_points, total_points);
    const double l = length_scale;
    const double sf2 = sigma_f * sigma_f;
    if (kernel_type == MaternType::Matern32) {
        const double sqrt3 = std::sqrt(3.0);
        Eigen::MatrixXd s = (sqrt3 * r.array() / l).matrix();
        K = sf2 * ((1.0 + s.array()) * (-s.array()).exp()).matrix();
    } else if (kernel_type == MaternType::Matern52) {
        const double sqrt5 = std::sqrt(5.0);
        Eigen::MatrixXd s = (sqrt5 * r.array() / l).matrix();
        Eigen::MatrixXd r2 = dist2;
        K = sf2 * ((1.0 + s.array() + 5.0 * r2.array() / (3.0 * l * l)) * (-s.array()).exp()).matrix();
    }

    // Add jitter
    K.diagonal().array() += 1e-6;

    // Cholesky decomposition (lower triangular)
    Eigen::LLT<Eigen::MatrixXd> llt(K.selfadjointView<Eigen::Lower>());
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("Cholesky decomposition failed");

    // Sample from standard normal
    std::normal_distribution<> N01(0.0, 1.0);
    Eigen::VectorXd z(total_points);
    for (int i = 0; i < total_points; ++i)
        z[i] = N01(gen);

    // Sample from GP
    Eigen::VectorXd f = llt.matrixL() * z;

    // Reshape to (M, N)
    Eigen::MatrixXd F(M, N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            F(i, j) = f(i * N + j);

    return F;
}


}