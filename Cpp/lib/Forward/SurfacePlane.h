#ifndef PLANE_SURFACE_H
#define PLANE_SURFACE_H

#include "Surface.h"
#include "../Utils/Constants.h"
#include <Eigen/Dense>

class SurfacePlane : public Surface {
public:
    SurfacePlane(Eigen::Vector3d cornerPoint,
        Eigen::Vector3d basis1,
        Eigen::Vector3d basis2,
        double size1,
        double size2,
        int resolution);

    const Eigen::MatrixXd& getPoints() const override;
    const Eigen::MatrixXd& getNormals() const override;
    const Eigen::MatrixXd& getTau1() const override;
    const Eigen::MatrixXd& getTau2() const override;


private:   
    Eigen::Vector3d cornerPoint_;
    Eigen::Vector3d basis1_;
    Eigen::Vector3d basis2_;
    double size1_;
    double size2_;
    int resolution_;

    Eigen::MatrixXd points_;
    Eigen::MatrixXd normals_;
    Eigen::MatrixXd tau1_;
    Eigen::MatrixXd tau2_;

    void generateSurface();
};

#endif // SPHERE_SURFACE_H
