#include "../../lib/Forward/MASSystem.h"
#include "../Utils/Constants.h"
#include "../Utils/UtilsPybind.h"
#include <map>
using std::ignore;


MASSystem::MASSystem(const py::object spline, const double lambda, const double dimension,
                        const Eigen::Vector3d& kinc, const Eigen::VectorXd& polarizations)
    : kinc_(kinc), polarizations_(polarizations)
    {
        generateSurface(spline, dimension);
    }





void MASSystem::generateSurface(py::object spline, double dimension){

    std::cout << "Generating surface" << std::endl;

    double maxcurvature = spline.attr("max_curvature").cast<double>();

    std::cout << "Max curvature: " << maxcurvature << std::endl;

    // --------- Generate test points -----------
    double lambda = constants.getWavelength();
    int test_point_res = static_cast<int>(std::ceil(constants.auxpts_pr_lambda * dimension/lambda)); // static_cast<int>(std::ceil(sqrt(2) * constants.auxpts_pr_lambda * dimension/lambda_));

    // Calculate points on surface and translate to Eigen
    auto result = PybindUtils::call_spline(spline, test_point_res);

    // Save in class
    this->points_ = result[0];
    this->normals_ = result[1];
    this->tau1_ = result[2];
    this->tau2_ = result[3];

    // Calculate control points on surface and translate to Eigen
    auto control_result = PybindUtils::call_spline(spline, 3*test_point_res);

    this->control_points_ = result[0]; // control_result[0];
    this->control_tangents1_ = result[2]; // control_result[2];
    this->control_tangents2_ = result[3]; // control_result[3];

    // --------- Generate auxiliary points -----------
    int aux_points_res = std::ceil(constants.auxpts_pr_lambda * dimension/lambda);

    // Calculate points on surface and translate to Eigen
    auto result_aux = PybindUtils::call_spline(spline, aux_points_res);
    
    // Calculate point values and save in class
    //radius = 1/max(maxcurvature, 1.0)
    double radius = 1.0 / std::max(maxcurvature, 1.0);

    bool radius1 = false;
    bool radius10 = true;

    if (radius1){
        radius = 1.0;
    } else if (radius10) {
        radius = 10.0;
    }

    std::cout << "Radius: " << radius << std::endl;

    this->aux_points_int_ = result_aux[0] - ((1 - constants.alpha) * radius) * result_aux[1]; 
    this->aux_points_ext_ = result_aux[0] + ((1 - constants.alpha) * radius) * result_aux[1];

    this->aux_normals_ = result_aux[1];
    this->aux_tau1_ = result_aux[2];
    this->aux_tau2_ = result_aux[3];

}