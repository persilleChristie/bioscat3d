#ifndef DIPOLE_H
#define DIPOLE_H

#include <Eigen/Dense>

class Dipole {
public:
    Dipole(const Eigen::Vector3d& pos, const Eigen::Vector3d& dir)
        : position_(pos), direction_(dir.normalized()) {}

    const Eigen::Vector3d& getPosition() const { return position_; }
    const Eigen::Vector3d& getDirection() const { return direction_; }

    void setPosition(const Eigen::Vector3d& pos) { position_ = pos; }
    void setDirection(const Eigen::Vector3d& dir) { direction_ = dir.normalized(); }

private:
    Eigen::Vector3d position_;  // Dipole center
    Eigen::Vector3d direction_; // Dipole orientation (unit vector)
};

#endif // DIPOLE_H
