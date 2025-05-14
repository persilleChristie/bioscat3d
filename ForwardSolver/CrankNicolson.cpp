#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <random>
#include "CrankNicolson.h"
#include "MASSystem.h"
#include "FieldCalculatorTotal.h"
#include "UtilsGP.h"

CrankNicolson::CrankNicolson(const Eigen::MatrixX3cd& data, 
                        const Eigen::MatrixX3d& data_points,
                        const double delta, 
                        const double gamma, 
                        const int iterations,
                        const char* jsonPath)
    : data_(data), data_points_(data_points), delta_(delta), 
    gamma_(gamma), iterations_(iterations), mas_("GP",jsonPath) {}

double CrankNicolson::logLikelihood(Eigen::ArrayXXd& points){
    mas_.setPoints(points.matrix());
    FieldCalculatorTotal field(mas_);

    int M = points.rows();

    Eigen::MatrixX3cd Eout = Eigen::MatrixX3cd::Zero(M, 3);    
    Eigen::MatrixX3cd Hout = Eigen::MatrixX3cd::Zero(M, 3);

    field.computeFields(Eout, Hout, data_points_);

    double loglike = -0.5 * gamma_ * (abs(Eout.col(2).array()) 
                                        - abs(data_.col(2).array())).matrix().squaredNorm();

    return loglike;
}

void CrankNicolson::run(){
    int M = data_.rows();

    Eigen::ArrayXXd previous = mas_.getPoints().array();
    Eigen::ArrayXXd proposal = mas_.getPoints().array();
    Eigen::MatrixXd gridpoints = previous.leftCols(2).matrix();

    double logLikelihoodPrev, logLikelihoodProp;
    double alpha;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double u;

    std::mt19937 genGP(std::random_device{}());

    double l = 0.2;
    double sigma = 0.2;

    // --------- Generate initial guess ---------
    previous.col(2) = GaussianProcess::sample_gp_2d_fast(gridpoints, l, sigma, genGP);
    logLikelihoodPrev = logLikelihood(previous);

    // --------- Run update ---------
    for (int i = 0; i < iterations_; ++i){
        // Draw samples
        proposal.col(2) = GaussianProcess::sample_gp_2d_fast(gridpoints, l, sigma, genGP);

        // Define new proposal
        proposal.col(2) = sqrt(1 - 2 * delta_) * previous.col(2) + sqrt(2 * delta_) * proposal.col(2);

        // Compute acceptance probability 
        logLikelihoodProp = logLikelihood(proposal);
        alpha = std::min(1.0, exp(logLikelihoodProp - logLikelihoodPrev));

        // Draw from uniform distribution
        u = distribution(generator);

        // Set f(i)
        if (u < alpha){
            previous = proposal;
            logLikelihoodPrev = logLikelihoodProp;
        } // else previous = previous, so nothing is done


        // ????? TODO: Update delta, gamma ?????
    }
}