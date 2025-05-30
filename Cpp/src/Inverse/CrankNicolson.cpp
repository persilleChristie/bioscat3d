#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <random>
#include "../../lib/Inverse/CrankNicolson.h"
#include "../../lib/Forward/MASSystem.h"
#include "../../lib/Forward/FieldCalculatorTotal.h"
#include "../../lib/Utils/UtilsGP.h"
#include "../../lib/Utils/UtilsExport.h"

CrankNicolson::CrankNicolson(const Eigen::MatrixX3cd& data, 
                        const Eigen::MatrixX3d& data_points,
                        const double delta, 
                        const double gamma, 
                        const int iterations,
                        const char* jsonPath)
    : data_(data), data_points_(data_points), delta_(delta), 
    gamma_(gamma), iterations_(iterations), mas_("GP", jsonPath) {
        std::cout << "pCN initialized succesfully" << std::endl;
    }

double CrankNicolson::logLikelihood(Eigen::MatrixXd& points){
    mas_.setPoints(points);
    FieldCalculatorTotal field(mas_);
    int M = data_points_.rows();

    Eigen::MatrixX3cd Eout = Eigen::MatrixX3cd::Zero(M, 3);    
    Eigen::MatrixX3cd Hout = Eigen::MatrixX3cd::Zero(M, 3);

    field.computeFields(Eout, Hout, data_points_);
    double loglike = -0.5 * gamma_ * (Eout.rowwise().squaredNorm()
                                        - data_.rowwise().squaredNorm()).squaredNorm();
    return loglike;
}

void CrankNicolson::run(bool verbose){
    Eigen::MatrixXd previous = mas_.getPoints();
    Eigen::MatrixXd proposal = mas_.getPoints();
    Eigen::MatrixXd gridpoints = previous.leftCols(2);
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

    int counter = 0;
    std::vector<int> accepted = {};
    std::vector<int> totalcount = {};

    // --------- Run update ---------
    for (int i = 0; i < iterations_; ++i){
        // Draw samples
        proposal.col(2) = GaussianProcess::sample_gp_2d_fast(gridpoints, l, sigma, genGP);

        // Define new proposal
        proposal.col(2) = (sqrt(1 - 2 * delta_) * previous.col(2).array() + sqrt(2 * delta_) * proposal.col(2).array()).matrix();

        // Compute acceptance probability 
        logLikelihoodProp = logLikelihood(proposal);
        alpha = std::min(1.0, exp(logLikelihoodProp - logLikelihoodPrev));

        // Draw from uniform distribution
        u = distribution(generator);
        // std::cout << "u: " << u << ", alpha: " << alpha << std::endl;

        // Set f(i)
        if (u < alpha){
            previous = proposal;
            logLikelihoodPrev = logLikelihoodProp;

            counter += 1;
            accepted.push_back(i);
        } // else previous = previous, so nothing is done

        totalcount.push_back(counter);
        // ????? TODO: Update delta, gamma ?????
    }

    // std::cout << "Surface: " << previous << std::endl;
    Export::saveRealVectorCSV("Estimate.csv", previous.col(2));
    
    std::ofstream file("totalcount.csv");
    if (file.is_open()) {
        for (size_t i = 0; i < totalcount.size(); ++i) {
            file << totalcount[i];
            if (i < totalcount.size() - 1)
                file << ",";
        }
        file << "\n";
        file.close();
        std::cout << "CSV file written successfully.\n";
    } else {
        std::cerr << "Unable to open file.\n";
    }

}