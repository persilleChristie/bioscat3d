#pragma once
#include <Eigen/Dense>
#include <random>
#include <optional>

struct SamplingPlan {
    Eigen::MatrixXd sourcePoints;
    Eigen::MatrixXd testPoints;
};

inline SamplingPlan createSamplingPlan(double lambda0, double Lx, double Ly,
                                       int pointsPerLambda, int testToSourceRatio,
                                       std::mt19937&,
                                       const std::optional<Eigen::MatrixXd>& surfaceNormals = std::nullopt) {
    const double dx = lambda0 / pointsPerLambda;
    const int nx_src = static_cast<int>(Lx / dx);
    const int ny_src = static_cast<int>(Ly / dx);
    const int N_src = nx_src * ny_src;

    // Test points on finer uniform grid
    const int nx_test = nx_src * testToSourceRatio;
    const int ny_test = ny_src * testToSourceRatio;
    const int N_test = nx_test * ny_test;
    const double dxt = Lx / nx_test;
    const double dyt = Ly / ny_test;

    Eigen::MatrixXd testPoints(N_test, 2);
    int index = 0;
    for (int i = 0; i < nx_test; ++i) {
        for (int j = 0; j < ny_test; ++j) {
            double x = -Lx / 2.0 + i * dxt;
            double y = -Ly / 2.0 + j * dyt;
            testPoints.row(index++) = Eigen::Vector2d(x, y);
        }
    }

    // Source points either vertically or along surface normals
    Eigen::MatrixXd sourcePoints(2 * N_test, 3);
    for (int i = 0; i < N_test; ++i) {
        const auto& xy = testPoints.row(i);
        Eigen::Vector3d base(xy(0), xy(1), 0.0);

        Eigen::Vector3d offset;
        if (surfaceNormals && surfaceNormals->rows() == N_test) {
            offset = (*surfaceNormals).row(i).normalized() * 2.0 * lambda0;
        } else {
            offset = Eigen::Vector3d(0.0, 0.0, 2.0 * lambda0);
        }

        sourcePoints.row(2 * i)     = base + offset;
        sourcePoints.row(2 * i + 1) = base - offset;
    }

    return SamplingPlan{sourcePoints, testPoints};
}
