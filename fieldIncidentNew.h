#ifndef _E_INCIDENT_NEW_H
#define _E_INCIDENT_NEW_H

#include <iostream>
#include <Eigen/Dense>


using namespace std;


pair<Eigen::Vector3cd, Eigen::Vector3cd> fieldIncidentNew(const Eigen::Vector3d& k_inc, const Eigen::Vector3cd& E_inc, const Eigen::Vector3d& x);


#endif