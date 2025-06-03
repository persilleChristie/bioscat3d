#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>


class CrankNicolson {
    public:
        CrankNicolson(const double dimension,
                        const Eigen::Vector3d& kinc, 
                        const double polarization,
                        const Eigen::MatrixX3cd& data, 
                        const Eigen::MatrixX3d& data_points,
                        const double delta, 
                        const double gamma, 
                        const int iterations); // Find a way to include covariance function?

        void run(bool verbose = false);

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
        const Eigen::MatrixX3cd data_;
        const Eigen::MatrixX3d data_points_;
        const double delta_;
        const double gamma_;
        const int iterations_;


        void constructor();
        double logLikelihood(Eigen::MatrixXd& points);

        
};