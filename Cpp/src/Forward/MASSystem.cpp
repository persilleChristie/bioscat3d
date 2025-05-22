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


std::vector<Eigen::MatrixX3d> call_spline(py::object spline, int resolution, bool control_points_flag){ 

    py::tuple result = spline.attr("calculate_points")(resolution, control_points_flag);

    py::array_t<double> py_points = result[0].cast<py::array_t<double>>();
    py::array_t<double> py_normals = result[1].cast<py::array_t<double>>();
    py::array_t<double> py_tangents1 = result[2].cast<py::array_t<double>>();
    py::array_t<double> py_tangents2 = result[3].cast<py::array_t<double>>();

    int rows = py_points.shape(0), cols = 3;

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    points((double*)py_points.request().ptr, rows, cols);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    normals((double*)py_normals.request().ptr, rows, cols);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    tangents1((double*)py_tangents1.request().ptr, rows, cols);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    tangents2((double*)py_tangents2.request().ptr, rows, cols);

    if (control_points_flag){
        py::array_t<double> py_control_points = result[4].cast<py::array_t<double>>();
        py::array_t<double> py_control_tangents1 = result[5].cast<py::array_t<double>>();
        py::array_t<double> py_control_tangents2 = result[6].cast<py::array_t<double>>();

        double rows_cp = py_control_points.shape(0);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        control_points((double*)py_control_points.request().ptr, rows_cp, cols);

        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        control_tangents1((double*)py_control_tangents1.request().ptr, rows_cp, cols);

        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        control_tangents2((double*)py_control_tangents2.request().ptr, rows_cp, cols);

        return {points, normals, tangents1, tangents2, control_points, control_tangents1, control_tangents2};

    }

    return {points, normals, tangents1, tangents2};

}



void MASSystem::generateSurface(py::object spline, double dimension){

    double maxcurvature = spline.attr("max_curvature").cast<double>();

    // --------- Generate test points -----------
    int test_point_res = static_cast<int>(std::ceil(sqrt(2) * constants.auxpts_pr_lambda * dimension/lambda_));

    // Calculate points on surface and translate to Eigen
    auto result = call_spline(spline, test_point_res, true);

    // Save in class
    this->points_ = result[0];
    this->tau1_ = result[2];
    this->tau2_ = result[3];
    this->control_points_ = result[4];
    this->control_tangents1_ = result[5];
    this->control_tangents2_ = result[6];

    // --------- Generate auxiliary points -----------
    int aux_points_res = std::ceil(constants.auxpts_pr_lambda * dimension/lambda_);

    // Calculate points on surface and translate to Eigen
    auto result_aux = call_spline(spline, aux_points_res, false);
    
    // Calculate point values and save in class
    this->aux_points_int_ = result[0] - constants.alpha * maxcurvature * result[1];   
    this->aux_points_ext_ = result[0] + constants.alpha * maxcurvature * result[1];

    this->aux_tau1_ = result[2];
    this->aux_tau2_ = result[3];

}