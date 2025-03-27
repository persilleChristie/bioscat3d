#ifndef DIPOLE_H
#define DIPOLE_H

#include <Eigen/Dense>

class Dipole {
public:
    Dipole(const Eigen::Vector3d& position, const Eigen::Vector3d& moment);

    const Eigen::Vector3d& getPosition() const;
    const Eigen::Vector3d& getMoment() const;

    void setPosition(const Eigen::Vector3d& pos);
    void setMoment(const Eigen::Vector3d& mom);

private:
    Eigen::Vector3d position;  // Dipole center
    Eigen::Vector3d moment;    // Dipole vector (direction × magnitude)
};

#endif // DIPOLE_H
