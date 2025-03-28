#include "SurfaceSphere.h"
#include "Constants.h"

using namespace Eigen;

SurfaceSphere::SurfaceSphere(double r, const Eigen::Vector3d& c, int res)
    : radius(r), center(c), resolution(res) {
    generateSurface();
}


const Eigen::MatrixXd& SurfaceSphere::getPoints() const  { return points;  }
const Eigen::MatrixXd& SurfaceSphere::getNormals() const { return normals; }
const Eigen::MatrixXd& SurfaceSphere::getTau1() const    { return tau1;    }
const Eigen::MatrixXd& SurfaceSphere::getTau2() const    { return tau2;    }

std::unique_ptr<Surface> SurfaceSphere::mirrored(const Vector3d& normal) const {
    auto mirroredSurface = std::make_unique<SurfaceSphere>(*this);
    mirroredSurface->points.col(2) *= -1;
    mirroredSurface->normals.col(2) *= -1;
    mirroredSurface->tau1.col(2) *= -1;
    mirroredSurface->tau2.col(2) *= -1;
    return mirroredSurface;
}

void SurfaceSphere::generateSurface() {
    Constants constants;
    int N = resolution;

    VectorXd theta0 = VectorXd::LinSpaced(N + 2, 0.0, constants.pi).segment(1, N);
    VectorXd phi0 = VectorXd::LinSpaced(N + 2, 0.0, 2 * constants.pi).segment(1, N);

    MatrixXd theta(N, N), phi(N, N);
    for (int i = 0; i < N; ++i) {
        theta.row(i) = theta0.transpose();
        phi.col(i) = phi0;
    }

    ArrayXXd st = theta.array().sin();
    ArrayXXd ct = theta.array().cos();
    ArrayXXd sp = phi.array().sin();
    ArrayXXd cp = phi.array().cos();

    ArrayXXd x = center(0) + radius * st * cp;
    ArrayXXd y = center(1) + radius * st * sp;
    ArrayXXd z = center(2) + radius * ct;

    int total = N * N;
    this->points.resize(total, 3);
    this->normals.resize(total, 3);
    this->tau1.resize(total, 3);
    this->tau2.resize(total, 3);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            Vector3d p(x(i, j), y(i, j), z(i, j));
            points.row(idx) = p;

            Vector3d n = (p - center).normalized();
            normals.row(idx) = n;

            Vector3d t1(radius * ct(i, j) * cp(i, j),
                        radius * ct(i, j) * sp(i, j),
                        -radius * st(i, j));
            tau1.row(idx) = t1.normalized();

            Vector3d t2(-radius * st(i, j) * sp(i, j),
                         radius * st(i, j) * cp(i, j),
                         0.0);
            tau2.row(idx) = t2.normalized();
        }
    }
}
