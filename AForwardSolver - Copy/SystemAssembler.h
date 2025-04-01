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
        Eigen::MatrixX3cd& A,
        Eigen::VectorXcd& b,
        const Surface& surface_mu,
        const std::vector<std::shared_ptr<FieldCalculator>>& sources,
        const std::shared_ptr<FieldCalculator>& incident
    );
};

#endif // SYSTEM_ASSEMBLER_H
