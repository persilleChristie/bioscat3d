#ifndef UTILSFRESNEL_H
#define UTILSFRESNEL_H

#include <cmath>
#include <complex>
#include <utility>

namespace UtilsFresnel {

inline std::pair<std::complex<double>, std::complex<double>> fresnelTE(
    double cos_theta_inc,
    double sin_theta_inc,
    double epsilon1,
    double epsilon2
) {
    std::complex<double> sqrt_term = std::sqrt(epsilon2 / epsilon1 * (1.0 - (epsilon1 / epsilon2) * sin_theta_inc * sin_theta_inc));
    std::complex<double> Gamma_r = (cos_theta_inc - sqrt_term) / (cos_theta_inc + sqrt_term);
    std::complex<double> Gamma_t = (2.0 * cos_theta_inc) / (cos_theta_inc + sqrt_term);
    return {Gamma_r, Gamma_t};
}

inline std::pair<std::complex<double>, std::complex<double>> fresnelTM(
    double cos_theta_inc,
    double sin_theta_inc,
    double epsilon1,
    double epsilon2
) {
    std::complex<double> sqrt_term = std::sqrt(epsilon1 / epsilon2 * (1.0 - (epsilon1 / epsilon2) * sin_theta_inc * sin_theta_inc));
    std::complex<double> Gamma_r = (-cos_theta_inc + sqrt_term) / (cos_theta_inc + sqrt_term);
    std::complex<double> Gamma_t = (2.0 * std::sqrt(epsilon1 / epsilon2) * cos_theta_inc) / (cos_theta_inc + sqrt_term);
    return {Gamma_r, Gamma_t};
}

} // namespace UtilsFresnel

#endif // UTILSFRESNEL_H
