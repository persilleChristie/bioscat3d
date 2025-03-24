#ifndef _COMPUTE_ANGLES_H
#define _COMPUTE_ANGLES_H

#include <iostream>
#include <Eigen/Dense>

using namespace std;


void computeAngles(const Eigen::Vector3d& x, double r, 
    double& cosTheta, double& sinTheta, 
    double& cosPhi, double& sinPhi);


#endif