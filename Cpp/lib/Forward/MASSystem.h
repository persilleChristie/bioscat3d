#ifndef MAS_SYSTEM_H
#define MAS_SYSTEM_H

#include "Constants.h"
#include <Eigen/Dense>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
    * Class MASSystem, storing a testsurface and two auxillariy surfaces for running MAS. It also stores all 
    * parameters for the incident field.
    *  
    * @param jsonPath char with path to json file including the parameters needed for each surface:
    *       - Bump needs 'resolution', 'halfWidth_x', 'halfWidth_y', 'halfWidth_z', 'bumpData' (hereunder 
    *                   'x0', 'y0', 'height' and 'sigma'), 'k' (incidence vector) and 'betas' (list of polarizations)
    *       - GP needs ...
    * @param surfaceType string indicating type of surface - allowed types are 'Bump' and 'GP'
    * @param constants updated Constants file
*/
class MASSystem {
public:
    MASSystem(const py::object spline, const double lambda, const double dimension, 
                const Eigen::Vector3d& kinc, const Eigen::VectorXd& polarizations); 

    // const std::vector<std::shared_ptr<FieldCalculatorUPW>>& getIncidentField();
    //const std::pair<Eigen::Vector3d, double>&getInc() const {return {kinc_, lambda_};};
    const Eigen::Vector3d getInc() const {return kinc_;};
    const Eigen::VectorXd& getPolarizations() const         {return polarizations_;};

    // void setPoints(Eigen::MatrixX3d points);
    const Eigen::MatrixX3d& getPoints() const   {return points_;};
    const Eigen::MatrixX3d& getNormals() const  {return normals_;};
    const Eigen::MatrixX3d& getTau1() const     {return tau1_;};
    const Eigen::MatrixX3d& getTau2() const     {return tau2_;};

    const Eigen::MatrixX3d& getIntPoints() const {return aux_points_int_;};
    const Eigen::MatrixX3d& getExtPoints() const {return aux_points_ext_;};
    const Eigen::MatrixX3d& getAuxNormals() const {return aux_normals_;};
    const Eigen::MatrixX3d& getAuxTau1() const {return aux_tau1_;};
    const Eigen::MatrixX3d& getAuxTau2() const {return aux_tau2_;};

    const Eigen::MatrixX3d& getControlPoints() const {return control_points_;};
    const Eigen::MatrixX3d& getControlTangents1() const {return control_tangents1_;};
    const Eigen::MatrixX3d& getControlTangents2() const {return control_tangents2_;};
    
    // const std::vector<int>& getIndecees() const {return aux_test_indices_;};

private:
    Eigen::Vector3d kinc_;
    Eigen::VectorXd polarizations_;

    Eigen::MatrixX3d points_;
    Eigen::MatrixX3d normals_;
    Eigen::MatrixX3d tau1_;
    Eigen::MatrixX3d tau2_;

    Eigen::MatrixX3d aux_points_int_;
    Eigen::MatrixX3d aux_points_ext_;
    Eigen::MatrixX3d aux_normals_;
    Eigen::MatrixX3d aux_tau1_;
    Eigen::MatrixX3d aux_tau2_;

    Eigen::MatrixX3d control_points_;
    Eigen::MatrixX3d control_tangents1_;
    Eigen::MatrixX3d control_tangents2_;
    // std::vector<int> aux_test_indices_;

    void generateSurface(py::object spline, double dimension);
};

#endif // MAS_SURFACE_H
