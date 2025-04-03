#pragma once
#include <Eigen/Dense>
#include <random>

struct SamplingPlan {
    Eigen::MatrixXd sourcePoints;
    Eigen::MatrixXd testPoints;
};

inline SamplingPlan createSamplingPlan(double lambda0, double Lx, double Ly,
                                       int pointsPerLambda, int testToSourceRatio,
                                       std::mt19937&) {
    const double dx = lambda0 / pointsPerLambda;
    const int nx_src = static_cast<int>(Lx / dx);
    const int ny_src = static_cast<int>(Ly / dx);
    const int N_src = nx_src * ny_src;

    // Source points on uniform grid
    Eigen::MatrixXd sourcePoints(N_src, 2);
    int index = 0;
    for (int i = 0; i < nx_src; ++i) {
        for (int j = 0; j < ny_src; ++j) {
            double x = -Lx / 2.0 + i * dx;
            double y = -Ly / 2.0 + j * dx;
            sourcePoints.row(index++) = Eigen::Vector2d(x, y);
        }
    }

    // Test points on finer uniform grid
    const int nx_test = nx_src * testToSourceRatio;
    const int ny_test = ny_src * testToSourceRatio;
    const int N_test = nx_test * ny_test;
    const double dxt = Lx / nx_test;
    const double dyt = Ly / ny_test;

    Eigen::MatrixXd testPoints(N_test, 2);
    index = 0;
    for (int i = 0; i < nx_test; ++i) {
        for (int j = 0; j < ny_test; ++j) {
            double x = -Lx / 2.0 + i * dxt;
            double y = -Ly / 2.0 + j * dyt;
            testPoints.row(index++) = Eigen::Vector2d(x, y);
        }
    }

    return SamplingPlan{sourcePoints, testPoints};
}
