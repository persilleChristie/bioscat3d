#ifndef BUMP_SURFACE_H
#define BUMP_SURFACE_H

#include "Surface.h"
#include "Constants.h"
#include <Eigen/Dense>

class SurfaceBumps : public Surface {
public:
    SurfaceBumps(const Eigen::VectorXd& x0,  const Eigen::VectorXd& y0, 
                 const Eigen::VectorXd& h, const Eigen::VectorXd& sigma, 
                 const double xdim, const double ydim, const double zdim, const int resolution,
                 Constants constants);

    const Eigen::MatrixXd& getPoints() const override  {return points_;};
    const Eigen::MatrixXd& getNormals() const override {return normals_;};
    const Eigen::MatrixXd& getTau1() const override    {return tau1_;};
    const Eigen::MatrixXd& getTau2() const override    {return tau2_;};

    const Eigen::MatrixXd& getInterior() const    {return aux_int_;};
    const Eigen::MatrixXd& getExterior() const    {return aux_ext_;};

    std::unique_ptr<Surface> mirrored(const Eigen::Vector3d& normal) const override;


private:
    Eigen::VectorXd x0_;  
    Eigen::VectorXd y0_;
    Eigen::VectorXd h_;
    Eigen::VectorXd sigma_;
    double xdim_;
    double ydim_;
    double zdim_;
    int resolution_;
    Constants constants_;
    int testpts_pr_lambda_ = 5;

    Eigen::MatrixXd points_;
    Eigen::MatrixXd normals_;
    Eigen::MatrixXd tau1_;
    Eigen::MatrixXd tau2_;
    Eigen::MatrixXd aux_int_;
    Eigen::MatrixXd aux_ext_;

    void generateSurface();
};

#endif // BUMP_SURFACE_H
