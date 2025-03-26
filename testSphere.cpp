#include <iostream>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include <list>

#include "linearSystem.h"
#include "Constants.h"
#include "matrixToCSVfile.h"

using namespace std;
using namespace Eigen;

struct Constants constants;

/*
tuple <MatrixXd, MatrixXd, MatrixXd, MatrixXd> sphere(const int num_points, const double radius, const Vector3d& center){
    double x0 = center(0), y0 = center(1), z0 = center(2);

    VectorXd theta0 = VectorXd::LinSpaced(num_points, constants.pi / (num_points + 1), constants.pi - constants.pi / (num_points + 1));
    VectorXd phi0 = VectorXd::LinSpaced(num_points, 2 * constants.pi / (num_points + 1), 2 * constants.pi - 2 * constants.pi / (num_points + 1));

    MatrixXd theta(num_points, num_points), phi(num_points, num_points);
    for (int i = 0; i < num_points; ++i) {
        theta.row(i) = VectorXd::Constant(num_points, theta0(i));
        phi.col(i) = phi0;
    }
    
    MatrixXd x = (x0 + radius * theta.array().sin() * phi.array().cos()).matrix();
    MatrixXd y = (y0 + radius * theta.array().sin() * phi.array().sin()).matrix();
    MatrixXd z = (z0 + radius * theta.array().cos()).matrix();
    
    MatrixXd points(num_points * num_points, 3);
    points << x.reshaped(1, num_points * num_points).transpose(),
          y.reshaped(1, num_points * num_points).transpose(),
          z.reshaped(1, num_points * num_points).transpose();
    
    MatrixXd normals = points.array().colwise() / points.colwise().norm().array();
    
    MatrixXd tau1(num_points * num_points, 3);
    tau1 << (radius * theta.array().cos() * phi.array().cos()).reshaped(1, num_points * num_points).transpose(),
        (radius * theta.array().cos() * phi.array().sin()).reshaped(1, num_points * num_points).transpose(),
        (-radius * theta.array().sin()).reshaped(1, num_points * num_points).transpose();
    
    tau1 = (tau1.array().colwise() / tau1.colwise().norm().array());
    
    MatrixXd tau2(num_points * num_points, 3);
    tau2 << (-radius * theta.array().sin() * phi.array().sin()).reshaped(1, num_points * num_points).transpose(),
        (radius * theta.array().sin() * phi.array().cos()).reshaped(1, num_points * num_points).transpose(),
        MatrixXd::Zero(num_points * num_points, 1);
    
    tau2 = (tau2.array().colwise() / tau2.colwise().norm().array());

    return {points, normals, tau1, tau2};
}
*/

int main(){

    // Print sphere data
    Vector3d center (0.0, 0.0, 0.0);
    // auto [points, normals, tau1, tau2] = sphere(10, 1, center);

    MatrixXd m = MatrixXd::Random(3,3);

    //dataMatrix << points, normals, tau1, tau2;

    list<string> colnames = {"t1", "t2", "t3"};//{"px", "py", "pz", "nx", "ny", "nz", "t1x", "t2x","t2y", "t2z"};

    matrixToCSVfile("test", colnames, m);

    return 0;
}
