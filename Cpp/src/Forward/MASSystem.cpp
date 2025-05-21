#include "../../lib/Forward/MASSystem.h"
#include "../Utils/Constants.h"
using std::ignore;


MASSystem::MASSystem(const py::object spline, const double lambda, const double dimension,
                        const Eigen::Vector3d& kinc, const Eigen::VectorXd& polarizations)
    : lambda_(lambda), kinc_(kinc), polarizations_(polarizations)
    {
        generateSurface(spline, dimension);
    }



// void MASSystem::setPoints(Eigen::MatrixX3d points) {
//     this->points_ = points;
// }

std::tuple<Eigen::MatrixX3d, Eigen::MatrixX3d, Eigen::MatrixX3d, Eigen::MatrixX3d, Eigen::MatrixX3d> 
        call_spline(py::object spline, int resolution){ 

    py::tuple result = spline.attr("calculate_points")(resolution);

    py::array_t<double> py_points = result[0].cast<py::array_t<double>>();
    py::array_t<double> py_normals = result[1].cast<py::array_t<double>>();
    py::array_t<double> py_tangents1 = result[2].cast<py::array_t<double>>();
    py::array_t<double> py_tangents2 = result[3].cast<py::array_t<double>>();
    py::array_t<double> py_control_points = result[4].cast<py::array_t<double>>();

    double rows = py_points.shape(0), cols = py_points.shape(1);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    points((double*)py_points.request().ptr, rows, cols);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    normals((double*)py_normals.request().ptr, rows, cols);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    tangents1((double*)py_tangents1.request().ptr, rows, cols);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    tangents2((double*)py_tangents2.request().ptr, rows, cols);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    control_points((double*)py_control_points.request().ptr, rows, cols);

    return {points, normals, tangents1, tangents2, control_points};

}



void MASSystem::generateSurface(py::object spline, double dimension){

    double maxcurvature = spline.attr("max_curvature").cast<double>();

    // --------- Generate test points -----------
    int test_point_res = static_cast<int>(std::ceil(sqrt(2) * constants.auxpts_pr_lambda * dimension/lambda_));

    // Calculate points on surface and translate to Eigen
    auto [points, normals, tangents1, tangents2, control_points] = call_spline(spline, test_point_res);

    // Save in class
    this->points_ = points;
    this->tau1_ = tangents1;
    this->tau2_ = tangents2;
    this->control_points_ = control_points;


    // --------- Generate auxiliary points -----------
    int aux_points_res = std::ceil(constants.auxpts_pr_lambda * dimension/lambda_);

    // Calculate points on surface and translate to Eigen
    auto [aux_points, aux_normals, aux_tangents1, aux_tangents2, aux_control_points] = call_spline(spline, aux_points_res);
    
    // Calculate point values and save in class
    this->aux_points_int_ = aux_points - constants.alpha * maxcurvature * aux_normals;   
    this->aux_points_ext_ = aux_points + constants.alpha * maxcurvature * aux_normals;

    this->aux_tau1_ = aux_tangents1;
    this->aux_tau2_ = aux_tangents2;

}