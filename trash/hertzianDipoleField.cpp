#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include "moveDipoleField.h"
#include "fieldDipole.h"
#include "computeAngles.h"
#include "hertzianDipoleField.h"


using namespace std;


pair<Eigen::Vector3cd, Eigen::Vector3cd> hertzianDipoleField(const Eigen::Vector3d& x,      // point to evaluate field in
                                                            const double Iel,               // "size" of Hertzian dipole
                                                            const Eigen::Vector3d& xPrime,  // Hertzian dipole placement
                                                            const Eigen::Vector3d& rPrime,  // Hertzian dipole direction
                                                            const double k0                 // wave constant
                                                            )
{
    Eigen::Vector3d x_origo = x - xPrime;
    double r = x_origo.norm();
    double cos_theta = 0.0, sin_theta = 0.0, cos_phi = 0.0, sin_phi = 0.0;
            
    computeAngles(x_origo, r, cos_theta, sin_theta, cos_phi, sin_phi);
            
    // Compute exp(-j * k0 * r) once
    complex<double> expK0r = polar(1.0, -k0 * r);

    auto [E_origo, H_origo] = fieldDipole(r, Iel, cos_theta, sin_theta, cos_phi, sin_phi, expK0r);

    auto [E, H] = moveDipoleField(xPrime, rPrime, E_origo, H_origo);

    return {E, H};
}

