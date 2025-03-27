#ifndef SYSTEMASSEMBLER_H
#define SYSTEMASSEMBLER_H

#include "Surface.h"
#include "FieldCalculator.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>

class SystemAssembler {
public:
    // Assemble system matrix A using a surface and collection of field sources
    static void assemble(
        Eigen::MatrixXd& A,
        const Surface& surface,
        const std::vector<std::shared_ptr<FieldCalculator>>& sources
    );
};

#endif // SYSTEMASSEMBLER_H