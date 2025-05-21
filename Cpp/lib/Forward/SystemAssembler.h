#ifndef SYSTEM_ASSEMBLER_H
#define SYSTEM_ASSEMBLER_H

#include "FieldCalculator.h"
#include <Eigen/Dense>
#include <memory>

class SystemAssembler {
public:
    static void assembleSystem(
        Eigen::MatrixXcd& A,
        Eigen::VectorXcd& b,
        Eigen::MatrixX3d points,
        Eigen::MatrixX3d tau1,
        Eigen::MatrixX3d tau2,
        const std::vector<std::shared_ptr<FieldCalculator>>& sources_int,
        const std::vector<std::shared_ptr<FieldCalculator>>& sources_ext,
        const std::shared_ptr<FieldCalculator>& incident
    );
};

#endif // SYSTEM_ASSEMBLER_H
