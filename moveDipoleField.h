#ifndef _MOVE_DIPOLE_FIELD_H
#define _MOVE_DIPOLE_FIELD_H

#include <iostream>
#include <Eigen/Dense>

using namespace std;


pair<Eigen::Vector3d, Eigen::Vector3d> moveDipoleField(
    const Eigen::Vector3d& xPrime,
    const Eigen::Vector3d& rPrime,
    const Eigen::Vector3cd& E,
    const Eigen::Vector3cd& H
);



#endif