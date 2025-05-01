#ifndef SYSTEM_ASSEMBLER_H
#define SYSTEM_ASSEMBLER_H

#include "Surface.h"
#include "FieldCalculator.h"
#include "Constants.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>


class SystemAssembler {
public:
    static void assembleSystem(
        Eigen::MatrixXcd& A,
        Eigen::VectorXcd& b,
        //const Surface& surface,
        Eigen::MatrixX3d points,
        Eigen::MatrixX3d tau1,
        Eigen::MatrixX3d tau2,
        const std::vector<std::shared_ptr<FieldCalculator>>& sources_int,
        // const std::vector<std::shared_ptr<FieldCalculator>>& sources_mirr,
        const std::vector<std::shared_ptr<FieldCalculator>>& sources_ext,
        const std::shared_ptr<FieldCalculator>& incident
        // const std::complex<double> Gamma_r
    );
};

#endif // SYSTEM_ASSEMBLER_H
