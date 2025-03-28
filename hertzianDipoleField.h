#ifndef _HERTZIAN_DIPOLE_FIELD_H
#define _HERTZIAN_DIPOLE_FIELD_H

#include <iostream>
#include <Eigen/Dense>

using namespace std;


pair<Eigen::Vector3cd, Eigen::Vector3cd> hertzianDipoleField(const Eigen::Vector3d& x,      // point to evaluate field in
                                                            const double Iel,               // "size" of Hertzian dipole
                                                            const Eigen::Vector3d& xPrime,  // Hertzian dipole placement
                                                            const Eigen::Vector3d& rPrime,  // Hertzian dipole direction
                                                            const double k0                 // wave constant
                                                            );


#endif