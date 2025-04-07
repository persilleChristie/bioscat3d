#ifndef SYSTEM_ASSEMBLER_H
#define SYSTEM_ASSEMBLER_H

#include "Surface.h"
#include "FieldCalculator.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>

class SystemAssembler {
public:
    static void assembleSystem(
        Eigen::MatrixXcd& A,
        Eigen::VectorXcd& b,
        const Surface& surface,
        const std::vector<std::shared_ptr<FieldCalculatorDipole>>& sources_int,
        const std::vector<std::shared_ptr<FieldCalculatorDipole>>& sources_mirr,
        const std::vector<std::shared_ptr<FieldCalculatorDipole>>& sources_ext,
        const std::shared_ptr<FieldCalculator>& incident,
        const std::complex<double> Gamma_r
    );
};

#endif // SYSTEM_ASSEMBLER_H
