#ifndef SPHERE_SURFACE_H
#define SPHERE_SURFACE_H

#include "Surface.h"
#include <Eigen/Dense>

class SurfaceSphere : public Surface {
public:
    SurfaceSphere(double radius, const Eigen::Vector3d& center, int resolution);

    const Eigen::MatrixXd& getPoints() const override;
    const Eigen::MatrixXd& getNormals() const override;
    const Eigen::MatrixXd& getTau1() const override;
    const Eigen::MatrixXd& getTau2() const override;

    std::unique_ptr<Surface> mirrored(const Eigen::Vector3d& normal) const override;

private:
    double radius;
    Eigen::Vector3d center;
    int resolution;

    Eigen::MatrixX3d points;
    Eigen::MatrixX3d normals;
    Eigen::MatrixX3d tau1;
    Eigen::MatrixX3d tau2;

    void generateSurface();
};

#endif // SPHERE_SURFACE_H
