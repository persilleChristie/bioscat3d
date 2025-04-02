#ifndef _SPHERE_PNT_H
#define _SPHERE_PNT_H

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


void sphere(
    double radius,
    const Vector3d& center,
    int num_points,
    MatrixXd& points,
    MatrixXd& normals,
    MatrixXd& tau1,
    MatrixXd& tau2
);


#endif