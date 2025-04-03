#pragma once
#include <Eigen/Dense>
#include <cmath>

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
