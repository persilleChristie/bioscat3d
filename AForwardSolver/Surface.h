#ifndef SURFACE_H
#define SURFACE_H

#include <Eigen/Dense>
#include <memory>

class Surface {
public:
    virtual ~Surface() = default;

    virtual const Eigen::MatrixXd& getPoints() const = 0;
    virtual const Eigen::MatrixXd& getNormals() const = 0;
    virtual const Eigen::MatrixXd& getTau1() const = 0;
    virtual const Eigen::MatrixXd& getTau2() const = 0;

    virtual std::unique_ptr<Surface> mirrored(const Eigen::Vector3d& normal) const = 0;
};

#endif // SURFACE_H
