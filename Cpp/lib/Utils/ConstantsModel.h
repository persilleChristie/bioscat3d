#ifndef CONSTANTS_MODEL_H
#define CONSTANTS_MODEL_H

#include <iostream>

/// @brief Global constants used in MAS model configuration.
struct ConstantsModel {
    /// @brief Default constructor (uses fallback defaults via setters)
    ConstantsModel() {
        setAuxPtsPrLambda(5);
        setAlpha(0.86);
    }

    /// @brief Parameterised constructor with runtime values
    ConstantsModel(int auxPts, double alphaValue) {
        setAuxPtsPrLambda(auxPts);
        setAlpha(alphaValue);
    }

    /// Setter for auxpts_pr_lambda
    void setAuxPtsPrLambda(int newAuxPts) {
        if (newAuxPts > 0) {
            auxpts_pr_lambda = newAuxPts;
        } else {
            std::cerr << "Error: auxpts_pr_lambda must be > 0. Resetting to 5.\n";
            auxpts_pr_lambda = 5;
        }
    }

    /// Getter for auxpts_pr_lambda
    int getAuxPtsPrLambda() const { return auxpts_pr_lambda; }

    /// Setter for alpha
    void setAlpha(double newAlpha) {
        if (newAlpha > 0) {
            alpha = newAlpha;
        } else {
            std::cerr << "Error: alpha must be > 0. Resetting to 0.86.\n";
            alpha = 0.86;
        }
    }

    /// Getter for alpha
    double getAlpha() const { return alpha; }


private:
    int auxpts_pr_lambda;
    double alpha;
};

// Global instance
extern ConstantsModel constantsModel;

#endif // CONSTANTS_MODEL_H
