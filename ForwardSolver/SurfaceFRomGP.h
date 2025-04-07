#pragma once
#include "Surface.h"
#include <Eigen/Dense>

class SurfaceFromGP : public Surface {
public:
    explicit SurfaceFromGP(const Eigen::MatrixXd& points);

    const Eigen::MatrixXd& getPoints() const override { return points_; }
    const Eigen::MatrixXd& getNormals() const override { return normals_; }
    const Eigen::MatrixXd& getTau1() const override { return tau1_; }
    const Eigen::MatrixXd& getTau2() const override { return tau2_; }

    std::unique_ptr<Surface> mirrored(const Eigen::Vector3d& normal) const override;

private:
    Eigen::MatrixXd points_;
    Eigen::MatrixXd normals_, tau1_, tau2_;

    void computeTangentsAndNormals();
};