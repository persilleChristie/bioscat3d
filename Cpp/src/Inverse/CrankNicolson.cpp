#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <random>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "../../lib/Inverse/CrankNicolson.h"
#include "../../lib/Forward/MASSystem.h"
#include "../../lib/Forward/FieldCalculatorTotal.h"
#include "../../lib/Utils/UtilsGP.h"
#include "../../lib/Utils/UtilsExport.h"
#include "../../lib/Utils/UtilsPybind.h"
#include "../../lib/Utils/Constants.h"
#include "../../lib/Utils/ConstantsModel.h"

namespace py = pybind11;


/// @brief Inititalizer for Crank Nicolson class
/// @param dimension size of sample (the sample is assumed to be quadratic)
/// @param lambda wavelength of incident wave
/// @param kinc incident wave propagation vector
/// @param polarization polarization of incident wave
/// @param data measurements taken in data points
/// @param data_points points of measurement
/// @param delta constant for pCN
/// @param gamma constant for pCN
/// @param iterations number of iterations of pCN
CrankNicolson::CrankNicolson(const double dimension, 
                        const Eigen::Vector3d& kinc, 
                        const double polarization,
                        const double power, 
                        const SurfacePlane& plane,
                        std::string kernel_type,
                        const double delta, 
                        const double gamma, 
                        const int iterations)
    : dimension_(dimension), kinc_(kinc), 
        polarization_(Eigen::VectorXd::Constant(1, polarization)),
        power_(power), plane_(plane), delta_(delta), 
        gamma_(gamma), iterations_(iterations) { 
        
        int fail = constructor(kernel_type);
        
        if (not fail){
            std::cout << "pCN initialized succesfully" << std::endl;
        }
    }

/// @brief Constructor for CrankNicolson class
/// This method generates the grid of points and initializes the Python spline class for further calculations.
int CrankNicolson::constructor(std::string kernel_type){

    if (kernel_type == "Matern"){
        this->SE_kernel_ = false;
    } else if (kernel_type == "SE") {
        this->SE_kernel_ = true;
    } else {
        std::cerr << "Wrong kernel type, allowed types are 'Matern' and 'SE'." << std::endl;
        return 1;
    }

    // --------- Generate grid ---------
    int N = static_cast<int>(std::ceil(sqrt(2) * constantsModel.getAuxPtsPrLambda() * dimension_ / constants.getWavelength()));

    double half_dim = dimension_ / 2.0;
    
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N, -half_dim, half_dim);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(N, -half_dim, half_dim);

    std::cout << "x: " << x.transpose() << std::endl;
    std::cout << "y: " << y.transpose() << std::endl;


    Eigen::MatrixXd X(N, N), Y(N, N);
    for (int i = 0; i < N; ++i) {  // Works as meshgrid
        X.row(i) = x.transpose();
        Y.col(i) = y;
    }

    this->X_ = X;
    this->Y_ = Y;

    // ------------ Run python code ---------------
    
    try {
        // Should this be scoped to ensure Python interpreter is started and stopped correctly?
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(1, ".");  // Add local dir to Python path
    
        // Import the Python module
        py::module spline_module = py::module::import("Spline");

        // Get the class
        this->SplineClass_ = std::move(spline_module.attr("Spline"));

        // Wrap Eigen matrices as NumPy arrays (shared memory, no copy)
        this->X_np_ = py::array_t<double>(PybindUtils::eigen2numpy(X));
        this->Y_np_ = py::array_t<double>(PybindUtils::eigen2numpy(Y));


    } catch (const py::error_already_set &e) {
        std::cerr << "Python error: " << e.what() << "\n";

        return 1;
    }
    
    return 0;
}

/// @brief Computes the log-likelihood of the data given the surface represented by Zvals.
/// @param Zvals The surface values at the grid points (NxM matrix).
/// @return The log-likelihood value.
/// @details This method uses the MASSystem and FieldCalculatorTotal classes to compute the fields at the data points
/// and calculates the log-likelihood based on the difference between the computed fields and the measured data.
/// @note The log-likelihood is computed as -0.5 * gamma * (Eout - data)^2, where Eout is the computed electric field at the data points.
double CrankNicolson::logLikelihood(Eigen::MatrixXd& Zvals){

    auto Z_np = PybindUtils::eigen2numpy(Zvals);

    py::object spline = SplineClass_(X_np_, Y_np_, Z_np);  

    MASSystem mas(spline, dimension_, kinc_, polarization_);
    
    FieldCalculatorTotal field(mas);

    Eigen::VectorXd power = field.computePower(plane_);

    double loglike = -0.5 * gamma_ * (power(0) - power_) * (power(0) - power_);
    
    return loglike;
}

/// @brief Runs the Crank-Nicolson algorithm to update the surface estimate.
/// @param verbose If true, prints additional information during the run.
/// @details This method initializes the Gaussian Process generator, generates an initial guess for the surface,
/// and iteratively updates the surface estimate using the pCN algorithm. It computes the acceptance probability
/// based on the log-likelihood of the proposed surface and the previous surface. If the proposal is accepted,
/// it updates the previous surface and log-likelihood; otherwise, it retains the previous surface.
/// The method also tracks the number of accepted proposals and can print this information if verbose is true.
void CrankNicolson::run(double l, double tau, int p, bool verbose){
    double logLikelihoodPrev, logLikelihoodProp;
    double alpha;

    // Initialize uniform generator
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double u;

    // Initialize Gaussian Process generator
    std::mt19937 genGP(std::random_device{}());

   
    // --------- Generate initial guess ---------
    Eigen::MatrixXd previous; 

    if (SE_kernel_){
        previous = GaussianProcess::sample_gp_on_grid_SE_fast(X_, Y_, l, tau, genGP);
    } else {
        previous = GaussianProcess::sample_gp_on_grid_matern_fast(X_, Y_, l, tau, p, genGP);
    }

    logLikelihoodPrev = logLikelihood(previous);

    Eigen::MatrixXd proposal = previous;

    int counter = 0;
    std::vector<int> accepted = {};
    std::vector<int> totalcount = {};

    // --------- Run update ---------
    for (int i = 0; i < iterations_; ++i){
        // Draw samples
        if (SE_kernel_){
            proposal = GaussianProcess::sample_gp_on_grid_SE_fast(X_, Y_, l, tau, genGP);
        } else {
            proposal = GaussianProcess::sample_gp_on_grid_matern_fast(X_, Y_, l, tau, p, genGP);
        }

        // Define new proposal
        proposal = sqrt(1 - 2 * delta_) * previous + sqrt(2 * delta_) * proposal;

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
        
    }

    if (verbose){
        std::cout << "Number of accepted proposals: " << totalcount[size(totalcount) - 1] << std::endl;
    }

    // // std::cout << "Surface: " << previous << std::endl;
    // Export::saveRealVectorCSV("Estimate.csv", previous.col(2));
    
    // std::ofstream file("totalcount.csv");
    // if (file.is_open()) {
    //     for (size_t i = 0; i < totalcount.size(); ++i) {
    //         file << totalcount[i];
    //         if (i < totalcount.size() - 1)
    //             file << ",";
    //     }
    //     file << "\n";
    //     file.close();
    //     std::cout << "CSV file written successfully.\n";
    // } else {
    //     std::cerr << "Unable to open file.\n";
    // }

}