#include <Eigen/Dense>
#include "../../lib/Forward/SurfacePlane.h"
#include "../../lib/Utils/Constants.h"

SurfacePlane::SurfacePlane(Eigen::Vector3d cornerPoint,
    Eigen::Vector3d basis1,
    Eigen::Vector3d basis2,
    double size1,
    double size2,
    int resolution
    ) : cornerPoint_(cornerPoint), basis1_(basis1), basis2_(basis2), size1_(size1), size2_(size2), resolution_(resolution)
    {
        generateSurface();
    }


const Eigen::MatrixXd& SurfacePlane::getPoints()  const { return points_;  }
const Eigen::MatrixXd& SurfacePlane::getNormals() const { return normals_;  }
const Eigen::MatrixXd& SurfacePlane::getTau1()    const { return tau1_;    }
const Eigen::MatrixXd& SurfacePlane::getTau2()    const { return tau2_;    }

std::unique_ptr<Surface> SurfacePlane::mirrored(const Eigen::Vector3d& normal) const {
    auto mirroredSurface = std::make_unique<SurfacePlane>(*this);
    mirroredSurface->points_.col(2) *= -1;
    mirroredSurface->normals_.col(2) *= -1;
    mirroredSurface->tau1_.col(2) *= -1;
    mirroredSurface->tau2_.col(2) *= -1;
    return mirroredSurface;
}



void SurfacePlane::generateSurface() {
    Constants constants;
    double totalsize1 = size1_ * basis1_.norm();
    double totalsize2 = size2_ * basis2_.norm();
    int N = resolution_;

    // Define mesh-grid
    Eigen::VectorXd multiplier1 = Eigen::VectorXd::LinSpaced(N, 0.0, totalsize1);
    Eigen::VectorXd multiplier2 = Eigen::VectorXd::LinSpaced(N, 0.0, totalsize2);

    Eigen::MatrixXd multipliers1(N, N), multipliers2(N, N);
    for (int i = 0; i < N; ++i) {
        multipliers1.row(i) = multiplier1.transpose();
        multipliers2.col(i) = multiplier2;
    }

    Eigen::ArrayXXd x = cornerPoint_(0) + multipliers1.array() * basis1_(0) + multipliers2.array() * basis2_(0);
    Eigen::ArrayXXd y = cornerPoint_(1) + multipliers1.array() * basis1_(1) + multipliers2.array() * basis2_(1);
    Eigen::ArrayXXd z = cornerPoint_(2) + multipliers1.array() * basis1_(2) + multipliers2.array() * basis2_(2);

    int total = N * N;
    this->points_.resize(total, 3);
    this->normals_.resize(total, 3);
    this->tau1_.resize(total, 3);
    this->tau2_.resize(total, 3);
    
    // Same normal and tangents for all points
    Eigen::Vector3d n = basis1_.cross(basis2_).normalized();
    Eigen::Vector3d basis1_n = basis1_.normalized();
    Eigen::Vector3d basis2_n = basis2_.normalized();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            Eigen::Vector3d p(x(i, j), y(i, j), z(i, j));
            points_.row(idx) = p;

            normals_.row(idx) = n;
            tau1_.row(idx) = basis1_n;
            tau2_.row(idx) = basis2_n;
        }
    }
}