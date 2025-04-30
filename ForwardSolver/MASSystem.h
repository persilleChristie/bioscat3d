#ifndef MAS_SYSTEM_H
#define MAS_SYSTEM_H

#include "Constants.h"
#include <Eigen/Dense>

class MASSystem {
public:
    MASSystem(const char* jsonPath, const std::string surfaceType, Constants constants);

    const Eigen::MatrixXd& getPoints() const   {return points_;};
    const Eigen::MatrixXd& getNormals() const  {return normals_;};
    const Eigen::MatrixXd& getTau1() const     {return tau1_;};
    const Eigen::MatrixXd& getTau2() const     {return tau2_;};

    const Eigen::MatrixXd& getInterior() const {return aux_int_;};
    const Eigen::MatrixXd& getExterior() const {return aux_ext_;};

private:
    Constants constants_;
    int testpts_pr_lambda_ = 5;

    Eigen::MatrixXd points_;
    Eigen::MatrixXd normals_;
    Eigen::MatrixXd tau1_;
    Eigen::MatrixXd tau2_;
    Eigen::MatrixXd aux_int_;
    Eigen::MatrixXd aux_ext_;

    void generateBumpSurface(const char* jsonPath);
    void generateGPSurface(const char* jsonPath);
};

#endif // MAS_SURFACE_H
