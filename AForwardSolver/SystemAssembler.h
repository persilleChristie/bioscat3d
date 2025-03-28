#ifndef SYSTEMASSEMBLER_H
#define SYSTEMASSEMBLER_H

#include "Surface.h"
#include "FieldCalculator.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>

class SystemAssembler {
public:
    // Assemble system matrix A and RHS vector b using a surface and field sources
    static void assemble(
        Eigen::MatrixXcd& A,
        Eigen::VectorXcd& b,
        const Surface& surface,
        const std::vector<std::shared_ptr<FieldCalculator>>& sources,
        const std::shared_ptr<FieldCalculator>& incidentField
    );
};

#endif // SYSTEMASSEMBLER_H