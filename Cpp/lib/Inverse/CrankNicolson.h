#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "../Forward/SurfacePlane.h"


class CrankNicolson {
    public:
        CrankNicolson(const double dimension,
                        const Eigen::Vector3d& kinc, 
                        const double polarization,
                        const double power, 
                        const SurfacePlane& plane,
                        const std::string kernel_type,
                        const double delta, 
                        const double gamma, 
                        const int iterations); // Find a way to include covariance function?

        Eigen::MatrixXd run(double l, double tau, int p = 0, bool verbose = false);

    private:
        // MAS constants
        const double dimension_;
        const Eigen::Vector3d kinc_;
        const Eigen::VectorXd polarization_;
        Eigen::MatrixXd X_;
        Eigen::MatrixXd Y_;
        pybind11::array_t<double> X_np_;
        pybind11::array_t<double> Y_np_;
        pybind11::object SplineClass_;

        // pCN
        const double power_;
        SurfacePlane plane_;
        const double delta_;
        const double gamma_;
        const int iterations_;
        bool SE_kernel_;


        int constructor(std::string kernel_type);
        double logLikelihood(Eigen::MatrixXd& points);

        
};