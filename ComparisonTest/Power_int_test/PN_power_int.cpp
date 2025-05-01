#include <Eigen/Dense>
#include <iostream>

#include "../../ForwardSolver/FieldCalculatorTotal.h"
#include "../../ForwardSolver/MASSystem.h"
#include "../../ForwardSolver/SurfacePlane.h"
#include "../../ForwardSolver/UtilsExport.h"
#include "../../ForwardSolver/Constants.h"

int main(){
    const char* jsonPath = "surfaceParams.json";

    MASSystem masSystem(jsonPath, "Bump", constants);

    FieldCalculatorTotal field(masSystem, constants);

    // Top
    double size1 = 2, size2 = 2;
    Eigen::Vector3d Cornerpoint (-1, -1, 10);
    Eigen::Vector3d basis1 (1,0,0), basis2 (0,1,0);
    SurfacePlane top(Cornerpoint, basis1, basis2, size1, size2, 20);
    Eigen::VectorXd powers = field.computePower(top);

    Export::saveRealVectorCSV("FilesCSV/powers_polarization.csv", powers);



    return 0;
}