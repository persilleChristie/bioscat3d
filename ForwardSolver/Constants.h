#ifndef _CONSTANTS_H
#define _CONSTANTS_H
//extern "C" {}

#include <complex>
#include <iostream>

struct Constants {

    // Mathematical constants
    static constexpr double pi = 3.14159265358979323846;

    // Imaginary unit (for complex exponentials etc.)
    static const inline std::complex<double> j = {0.0, 1.0}; // Imaginary unit

    // Physical constants
    static constexpr double eta0     = 1; // 377.0;            // 377 Ohm, impedance of free space
    static constexpr double n0       = 1.0;              // Refractive index of air
    static constexpr double epsilon0 = 1; // 8.8541878188e-12; // Permittivity of free space
    static constexpr double epsilon1 = 1.0;              // Substrate epsilon value????
    static constexpr double mu0      = 1; // 1.25663706127e-6; // Permeability of free space
    static constexpr double Iel      = 1.0;              // Source current (default value)

    // Chosen constants (I have used the same constants as Mirza for some of them)
    static constexpr double n1    = 1.6;         // Refractive index of the nanowire/nanostructure
    static constexpr double alpha = 0.86;        // Scaling for placement of auxiliary sources for the nanowires

    // Computed constants
    double k0;              // Wave number in free space
    double k1;              // Wave number in nanowire/nanostructure

    // Constructor
    Constants() {
        // Computed constants
        setWavelength(1.0); // setWavelength(325e-9); // Default value of wavelength in free space
        //printf("Constants initialized!\n");
    }

    // Change wavelength and update wave numbers
    void setWavelength(double lambda) {
        lambda0 = lambda;
        k0 = 2 * pi / lambda0;         // Wave number in free space
        k1 = k0 * n1 / n0;             // Wave number in nanowire/nanostructure
        //printf("Constants updated with new wavelength!\n");
    }

    // Read current wavelength
    double getWavelength() const { return lambda0; }

private:
    double lambda0;         // Wavelength in free space (private to enforce consistent updates)
};

// Global instance
extern Constants constants;

#endif
