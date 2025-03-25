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
    points << x.reshaped(num_points * num_points, 1),
              y.reshaped(num_points * num_points, 1),
              z.reshaped(num_points * num_points, 1);
    
    MatrixXd normals = points.array().rowwise() / points.rowwise().norm().array();
    
    MatrixXd tau1(num_points * num_points, 3);
    tau1 << radius * theta.array().cos().reshaped(num_points * num_points, 1) * phi.array().cos().reshaped(num_points * num_points, 1),
            radius * theta.array().cos().reshaped(num_points * num_points, 1) * phi.array().sin().reshaped(num_points * num_points, 1),
            -radius * theta.array().sin().reshaped(num_points * num_points, 1);
    
    tau1 = (tau1.array().rowwise() / tau1.rowwise().norm().array());
    
    MatrixXd tau2(num_points * num_points, 3);
    tau2 << -radius * theta.array().sin().reshaped(num_points * num_points, 1) * phi.array().sin().reshaped(num_points * num_points, 1),
            radius * theta.array().sin().reshaped(num_points * num_points, 1) * phi.array().cos().reshaped(num_points * num_points, 1),
            MatrixXd::Zero(num_points * num_points, 1);
    
    tau2 = (tau2.array().rowwise() / tau2.rowwise().norm().array());

    return {points, normals, tau1, tau2};
}

int main(){

    // Print sphere data
    Vector3d center (0.0, 0.0, 0.0);
    auto [points, normals, tau1, tau2] = sphere(10, 1, center);

    MatrixXd dataMatrix(10*10, 3*4);

    dataMatrix << points, normals, tau1, tau2;

    list<string> colnames = {"px", "py", "pz", "nx", "ny", "nz", "t1x", "t2x","t2y", "t2z"};

    matrixToCSVfile("testSphere", colnames, dataMatrix);

    return 0;
}
