#include "Dipole.h"

Dipole::Dipole(const Eigen::Vector3d& position, const Eigen::Vector3d& moment)
    : position(position), moment(moment) {}

const Eigen::Vector3d& Dipole::getPosition() const {
    return position;
}

const Eigen::Vector3d& Dipole::getMoment() const {
    return moment;
}

void Dipole::setPosition(const Eigen::Vector3d& pos) {
    position = pos;
}

void Dipole::setMoment(const Eigen::Vector3d& mom) {
    moment = mom;
}
