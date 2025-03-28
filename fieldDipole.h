#ifndef _FIELD_DIPOLE_H
#define _FIELD_DIPOLE_H

#include <iostream>
#include <Eigen/Dense>

using namespace std;


pair<Eigen::Vector3cd, Eigen::Vector3cd> fieldDipole(double r, double Iel, 
    double cos_theta, double sin_theta, double cos_phi, double sin_phi, 
    const complex<double>& expK0r);


#endif