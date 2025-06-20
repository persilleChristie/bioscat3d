#ifndef GAUSSIAN_PROCESS_H
#define GAUSSIAN_PROCESS_H

// #pragma once
#include <Eigen/Dense>
#include <cmath>
#include <random>

/// @brief Namespace for Gaussian Process utilities.
/// @details This namespace contains functions for Gaussian Process kernels, mean functions, and sampling methods.
namespace GaussianProcess{

/// @brief Samples from a Gaussian Process with SE kernel on a grid defined by X1 and X2.
/// @param X1 First dimension grid points (M, N).
/// @param X2 Second dimension grid points (M, N).
/// @param l Length scale parameter for the RBF kernel.
/// @param tau Signal variance parameter for the RBF kernel.
/// @param gen Random number generator.
/// @return A matrix of shape (M, N) containing the sampled values from the Gaussian Process on the grid.
/// @details This function flattens the meshgrid defined by X1 and X2 into a single matrix of shape (total_points, 3),
/// computes the pairwise squared distances, applies the RBF kernel, and samples from the Gaussian Process using Cholesky decomposition.
/// The output is reshaped back to the original grid shape (M, N).
inline Eigen::MatrixXd sample_gp_on_grid_SE_fast(const Eigen::MatrixXd& X1,
                                                  const Eigen::MatrixXd& X2,
                                                  double l,
                                                  double tau,
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
    Eigen::MatrixXd K = (- 1 / (l * l) * dist2.array()).exp().matrix() * tau * tau;

    // Add jitter for numerical stability
    K.diagonal().array() += 1e-6;

    // Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> llt(K.selfadjointView<Eigen::Lower>());
    if (llt.info() != Eigen::Success){
        throw std::runtime_error("Cholesky decomposition failed");
    }


    // Sample standard normal vector
    std::normal_distribution<> N01(0.0, 1.0);
    Eigen::VectorXd z(total_points);
    for (int i = 0; i < total_points; ++i){
        z[i] = N01(gen);
    }

    // Sample GP and reshape
    Eigen::VectorXd f = llt.matrixL() * z;
    Eigen::MatrixXd F(M, N);
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            F(i, j) = f(i * N + j);
        }   
    }

    return F;
}


/// @brief Samples from a Gaussian Process with Matern kernel on a grid defined by X1 and X2.
/// @param X1 First dimension grid points (M, N).
/// @param X2 Second dimension grid points (M, N).
/// @param l Length scale parameter for the Matern kernel.
/// @param tau Signal variance parameter for the Matern kernel.
/// @param p Integer value of p controlling nu
/// @param gen Random number generator.
/// @return A matrix of shape (M, N) containing the sampled values from the Gaussian Process on the grid.
/// @details This function flattens the meshgrid defined by X1 and X2 into a single matrix of shape (total_points, 2),
/// computes the pairwise squared distances, applies the selected Matern kernel, and samples from the Gaussian Process using Cholesky decomposition.
/// The output is reshaped back to the original grid shape (M, N).
inline Eigen::MatrixXd sample_gp_on_grid_matern_fast(const Eigen::MatrixXd& X1,
                                                     const Eigen::MatrixXd& X2,
                                                     double l,
                                                     double tau,
                                                     int p,
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

    Eigen::ArrayXXd kappa = (std::sqrt(2 * p + 1) * r.array() / l);

    Eigen::ArrayXXd sum = Eigen::ArrayXXd::Zero(total_points, total_points);

    for (int i = 0; i <= p; ++i){
        sum += tgamma(p + i + 1) / (tgamma(i + 1) * tgamma(p - i +1)) * (2 * kappa).pow(p - i);
    }

    
    K = tau * tau * (sum * tgamma(p + 1)/tgamma(2*p + 1) * (-kappa).exp()).matrix();

    // Add jitter
    K.diagonal().array() += 1e-6;

    // Cholesky decomposition (lower triangular)
    Eigen::LLT<Eigen::MatrixXd> llt(K.selfadjointView<Eigen::Lower>());
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("Cholesky decomposition failed");

    // Sample from standard normal
    std::normal_distribution<> N01(0.0, 1.0);
    Eigen::VectorXd z(total_points);
    for (int i = 0; i < total_points; ++i){
        z[i] = N01(gen);
    }

    // Sample from GP
    Eigen::VectorXd f = llt.matrixL() * z;

    // Reshape to (M, N)
    Eigen::MatrixXd F(M, N);
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            F(i, j) = f(i * N + j);
        }   
    }

    return F;
}

} // namespace GaussianProcess
#endif // GAUSSIAN_PROCESS_H



// // --- RBF Kernel ---
// // k(x, x') = σ_f² * exp(-||x - x'||² / (2ℓ²))
// /// @brief Computes the Radial Basis Function (RBF) kernel between two points.
// /// @param x1 First point as an Eigen vector.
// /// @param x2 Second point as an Eigen vector.
// /// @param lengthScale Length scale parameter for the kernel.
// /// @param sigmaF Signal variance parameter for the kernel.
// /// @return The value of the RBF kernel between x1 and x2.
// inline double rbf_kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
//                          double lengthScale, double sigmaF) {
//     double r2 = (x1 - x2).squaredNorm();
//     return sigmaF * sigmaF * std::exp(-r2 / (2.0 * lengthScale * lengthScale));
// }

// // --- Matern 3/2 Kernel ---
// // k(x, x') = σ_f² * (1 + sqrt(3) r / ℓ) * exp(-sqrt(3) r / ℓ)
// /// @brief Computes the Matern 3/2 kernel between two points.
// /// @param x1 First point as an Eigen vector.
// /// @param x2 Second point as an Eigen vector.
// /// @param lengthScale Length scale parameter for the kernel.
// /// @param sigmaF Signal variance parameter for the kernel.
// /// @return The value of the Matern 3/2 kernel between x1 and x2.
// inline double matern32_kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
//                               double lengthScale, double sigmaF) {
//     double r = (x1 - x2).norm();
//     double s = std::sqrt(3.0) * r / lengthScale;
//     return sigmaF * sigmaF * (1.0 + s) * std::exp(-s);
// }

// // --- Matern 5/2 Kernel ---
// // k(x, x') = σ_f² * (1 + sqrt(5) r / ℓ + 5r²/(3ℓ²)) * exp(-sqrt(5) r / ℓ)
// /// @brief Computes the Matern 5/2 kernel between two points.
// /// @param x1 First point as an Eigen vector.
// /// @param x2 Second point as an Eigen vector.
// /// @param lengthScale Length scale parameter for the kernel.
// /// @param sigmaF Signal variance parameter for the kernel.
// /// @return The value of the Matern 5/2 kernel between x1 and x2.
// inline double matern52_kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2,
//                               double lengthScale, double sigmaF) {
//     double r = (x1 - x2).norm();
//     double s = std::sqrt(5.0) * r / lengthScale;
//     return sigmaF * sigmaF * (1.0 + s + (5.0 / 3.0) * r * r / (lengthScale * lengthScale)) * std::exp(-s);
// }

// // --- Zero Mean Function ---
// /// @brief Returns a zero mean vector for Gaussian Processes.
// /// @param X Input matrix (not used, but can be used to determine the size of the output).
// /// @return A zero vector of size equal to the number of rows in X.
// inline Eigen::VectorXd zero_mean(const Eigen::MatrixXd& X) {
//     return Eigen::VectorXd::Zero(X.rows());
// }

// /// @brief Samples from a Gaussian Process with RBF kernel.
// /// @param X Input matrix of shape (N, D) where N is the number of points and D is the dimensionality.
// /// @param length_scale Length scale parameter for the RBF kernel.
// /// @param sigma_f Signal variance parameter for the RBF kernel.
// /// @param gen Random number generator.
// /// @return A vector of shape (N,) containing the sampled values from the Gaussian Process.
// inline Eigen::VectorXd sample_gp_2d_fast(const Eigen::MatrixXd& X,
//                                   double length_scale,
//                                   double sigma_f,
//                                   std::mt19937& gen) {
//     const int N = X.rows();
//     const double inv_2l2 = 0.5 / (length_scale*length_scale);
//     const double sigmaf2     = sigma_f * sigma_f;
//     const double jitter  = 1e-6;

//     // 1) Distance matrix via BLAS
//     Eigen::VectorXd sqnorm = X.rowwise().squaredNorm();
//     Eigen::MatrixXd G      = X * X.transpose();
//     Eigen::MatrixXd dist2  = sqnorm.replicate(1, N)
//                            + sqnorm.transpose().replicate(N, 1)
//                            - 2.0 * G;

//     // 2) Covariance
//     Eigen::MatrixXd K = (-inv_2l2 * dist2.array())
//                         .exp()
//                         .matrix() * sigmaf2;
//     K.diagonal().array() += jitter;

//     // 3) Cholesky on self-adjoint view
//     Eigen::LLT<Eigen::MatrixXd> llt(K.selfadjointView<Eigen::Lower>());
//     if (llt.info() != Eigen::Success)
//         throw std::runtime_error("Cholesky failed");

//     // 4) Sample z ~ N(0,I)
//     std::normal_distribution<> N01(0.0,1.0);
//     Eigen::VectorXd z(N);
//     for (int i = 0; i < N; ++i) z[i] = N01(gen);

//     // 5) Return L * z
//     return llt.matrixL() * z;
// }