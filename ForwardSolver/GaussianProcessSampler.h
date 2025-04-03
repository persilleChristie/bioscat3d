#pragma once
#include "GaussianProcessModel.h"
#include "Surface.h"
#include <random>

class GaussianProcessSampler {
public:
    GaussianProcessSampler(const GaussianProcessModel& gp_model);

    // Returns a Surface constructed from a sample
    std::unique_ptr<Surface> samplePosteriorSurface(const Eigen::MatrixXd& X_test);
    std::unique_ptr<Surface> samplePriorSurface(const Eigen::MatrixXd& X_test);

private:
    const GaussianProcessModel& gp_;
    mutable std::mt19937 rng_;
};
