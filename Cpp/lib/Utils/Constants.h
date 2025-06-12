#ifndef CONSTANTS_H
#define CONSTANTS_H
//extern "C" {}

#include <complex>
#include <iostream>
#include <stdio.h>

/// Constants.h
/// @file Constants.h
/// @brief This file defines a Constants class that holds various physical and mathematical constants.
/// @details The Constants class includes constants such as the speed of light, permittivity and permeability of free space,
/// refractive indices, and the impedance of free space. It also provides methods to set and get the wavelength,
/// and computes the wave numbers in free space and in a nanowire/nanostructure based on the current wavelength.
/// @note The class is designed to be used in electromagnetic simulations, particularly in the context of nanowire optics.
struct Constants {
    // Mathematical constants
    static constexpr double pi = 3.14159265358979323846;

    // Imaginary unit (for complex exponentials etc.)
    static constexpr inline std::complex<double> j = {0.0, 1.0}; // Imaginary unit

    // Physical constants
    static constexpr double n0       = 1.0;              // Refractive index of air
    static constexpr double epsilon0 = 1; // Normalized permittivity of free space (in arbitrary units) // 8.8541878188e-12; // Permittivity of free space
    static constexpr double epsilon1 = 2.56;             // Substrate epsilon value????
    static constexpr double mu0      = 1; // Normalized permeability of free space (in arbitrary units) // 1.25663706127e-6; // Permeability of free space
    static constexpr double Iel      = 1.0;              // Source current (default value)

    static constexpr double eta0 = sqrt(mu0/epsilon0); // 377.0;            // 377 Ohm, impedance of free space
    static constexpr double eta1 = sqrt(mu0/epsilon1);                // Impedance of the nanowire/nanostructure

    // Chosen constants (I have used the same constants as Mirza for some of them)
    static constexpr double n1   = sqrt(epsilon1/epsilon0); // 1.6;         // Refractive index of the nanowire/nanostructure 
                                                                 // (relative permittivity times relative permeability)
    static constexpr double alpha = 0.86;        // Scaling for placement of auxiliary sources for the nanowires

    static constexpr int auxpts_pr_lambda = 5;

    // Computed constants
    double k0;              // Wave number in free space
    double k1;              // Wave number in nanowire/nanostructure

    // Constructor
    /// @brief Default constructor for Constants.
    /// @details Initializes the Constants object with default values for wavelength and computes the wave numbers.
    /// The default wavelength is set to 325 nm, which is a common value in optics.
    /// @note The constructor also computes the wave numbers in free space (k0) and in the nanowire/nanostructure (k1)
    /// based on the default wavelength.
    Constants() {
        // Computed constants
        setWavelength(325e-9); // Default value of wavelength in free space
        // printf("Constants initialized!\n");
    }

    // Change wavelength and update wave numbers
    /// @brief Set a new wavelength and update the wave numbers accordingly.
    /// @param lambda New wavelength in meters.
    /// @details This method updates the wavelength and recalculates the wave numbers in free space (k0)
    /// and in the nanowire/nanostructure (k1) based on the new wavelength.
    void setWavelength(double lambda) {
        lambda0 = lambda;
        k0 = 2 * pi / lambda0;         // Wave number in free space
        k1 = k0 * n1 / n0;             // Wave number in nanowire/nanostructure
        // printf("Constants updated with new wavelength!\n");
    }

    // Read current wavelength
    /// @brief Get the current wavelength in free space.
    /// @return The current wavelength.
    double getWavelength() const { return lambda0; }

private:
    double lambda0;         // Wavelength in free space (private to enforce consistent updates)
};

// Global instance
extern Constants constants;

#endif // CONSTANTS_H
