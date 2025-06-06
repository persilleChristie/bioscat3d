#ifndef MAS_SYSTEM_H
#define MAS_SYSTEM_H

#include "Constants.h"
#include <Eigen/Dense>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/// @class MASSystem
/// @brief Represents a Method of Auxiliary Sources (MAS) system including the surface and incident field.
/// @details
/// Stores and manages the test surface and auxiliary points (interior and exterior),
/// generated from a Python spline. Also stores the parameters for the incident field,
/// such as the wavevector and polarization angles.
class MASSystem {
public:
    /// @brief Constructor that initializes the MAS surface system.
    /// @param spline A Python object representing the spline surface (must expose `max_curvature` and be callable).
    /// @param dimension The physical dimension (width) of the surface domain.
    /// @param kinc The incident wave vector as an Eigen::Vector3d.
    /// @param polarizations The polarizations of the incident field as an Eigen::VectorXd.
    MASSystem(const py::object spline, const double dimension, 
                const Eigen::Vector3d& kinc, const Eigen::VectorXd& polarizations); 

    // const std::vector<std::shared_ptr<FieldCalculatorUPW>>& getIncidentField();
    // const std::pair<Eigen::Vector3d, double>&getInc() const {return {kinc_, lambda_};};
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

    /// @brief Generates the surface points, normals, and tangents based on the provided spline object.
    /// @param spline A Python object representing the spline surface.
    /// @param dimension The dimension of the MAS system.
    void generateSurface(py::object spline, const double dimension);
};

#endif // MAS_SYSTEM_H
