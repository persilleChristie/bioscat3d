#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include "MASSystem.h"

class CrankNicolson {
    public:
        CrankNicolson(const Eigen::MatrixX3cd& data, 
                        const Eigen::MatrixX3d& data_points,
                        const double delta, 
                        const double gamma, 
                        const int iterations,
                        const char* jsonPath); // Find a way to include covariance function?

        void run();

    private:
        const Eigen::MatrixX3cd data_;
        const Eigen::MatrixX3d data_points_;
        const double delta_;
        const double gamma_;
        const int iterations_;
        MASSystem mas_;
        // const char * jsonPath_;

        void constructor();
        double logLikelihood(Eigen::MatrixXd& points);

        
};