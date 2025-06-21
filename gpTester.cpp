#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include "Cpp/lib/Utils/UtilsGP.h"
#include "Cpp/lib/Utils/UtilsExport.h"

int main(){
    
    Eigen::Vector4i lvalues(1, 2, 3, 4);
    Eigen::Vector3d tauvalues(0.25, 0.5, 1);
    Eigen::Vector4i pvalues(1, 2, 3, 4);

    // Create grid
    int N = 50;
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N, 0, 5);
    Eigen::MatrixXd X = x.replicate(1, N);           
    Eigen::MatrixXd Y = x.transpose().replicate(N, 1); 

    // Initialize Gaussian Process generator
    std::mt19937 genGP(std::random_device{}());

    for (int i = 0; i < 5; ++i){

        std::cout << "----------------- Running for i = " << i << " --------------------" << std::endl;
        // Sample GP for squared exponential
        for (auto l : lvalues){
            for (auto tau : tauvalues){
                auto sample = GaussianProcess::sample_gp_on_grid_SE_fast(X, Y, l, tau, genGP);

                int tau_int = static_cast<int>(std::round(tau * 100));
                Export::saveRealMatrixCSV("../CSV/GPsamples/SE_sample" + std::to_string(i) +  "_l_" + std::to_string(l) 
                                            + "_tau_" + std::to_string(tau_int) + ".csv", sample);


            }
        }

        std::cout << "Saved all SE" << std::endl;
        
        // Sample GP for Matern
        for (auto l : lvalues){
            for (auto tau : tauvalues){
                int tau_int = static_cast<int>(std::round(tau * 100));

                for (auto p : pvalues){
                    auto sample = GaussianProcess::sample_gp_on_grid_matern_fast(X, Y, l, tau, p, genGP);

                    Export::saveRealMatrixCSV("../CSV/GPsamples/Matern_sample" + std::to_string(i) +  "_l_" + std::to_string(l) 
                                            + "_tau_" + std::to_string(tau_int) + "_p_" + std::to_string(p) + ".csv", sample);

                }
            }
        }
        
        std::cout << "Saved all Matern" << std::endl;
    
    }




    return 0;
}