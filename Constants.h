#ifndef _CONSTANTS_H
#define _CONSTANTS_H
//extern "C" {

struct Constants {

    // Mathematical constants
    static constexpr double pi = 3.14159265358979323846;

    // Physical constants
    static constexpr double eta0     = 377.0;            // 377 Ohm, impedance of free space
    static constexpr double n0       = 1.0;              // Refractive index of air
    static constexpr double epsilon0 = 8.8541878188e-12; // Permittivity of free space
    static constexpr double mu0      = 1.25663706127e-6; // Permeability of free space

    // Chosen constants (I have used the same constants as Mirza for some of them)
    static constexpr double lambda0 = 325e-9;    // Default value of wavelength in free space
    static constexpr double n1    = 1.6;         // Refractive index of the nanowire/nanostructure % 5.05506 - 1i*3.20418;
    static constexpr double alpha = 0.86;        // Scaling for placement of auxiliary sources for the nanowires

    // Computed constants
    double k0;              // Wave number in free space
    double k1;              // Wave number in nanowire/nanostructure

    Constants() {
        // Computed constants
        k0          = 2*pi/lambda0;         // Wave number in free space
        k1          = k0*n1/n0;             // Wave number in nanowire/nanostructure
        //printf("Constants initialized!\n");
    }

    /*
    void newWavelength(double lambda) {
        lambda0 = lambda;
        // Computed constants
        k0          = 2*pi/lambda0;         // Wave number in free space
        k1          = k0*n1/n0;             // Wave number in nanowire/nanostructure
        //printf("Constants initialized with new wavelength!\n");
    }
        */


};

//}
#endif
