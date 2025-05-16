#include "../ForwardSolver/FieldCalculatorDipole.h"
#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include "../ForwardSolver/Constants.h"
#include "../ForwardSolver/UtilsDipole.h"

std::complex<double> j = {0.0, 1.0};
Constants constants;

int main(){
    // Define tests from maple
    Eigen::Vector3d r (1,2,3);

    Eigen::Vector3d x0(1,1,1);

    Eigen::Matrix3d xs;
    xs << 4,7,10,
        -2,3,100,
        -200,700,1100;

    Eigen::Matrix3cd unitTestsE;
    unitTestsE << -0.2467695595e8 + 0.2830407954e8*j, -0.8769011342e7 + 0.1005791953e8*j, 0.1407166196e8 - 0.1613997073e8*j,
                -0.1593631571e7 + 0.2950742015e7*j, -46021.3958 + 85212.4815*j, -47361.7032 + 87695.195*j,
                -203984.1370 - 57791.70737*j, -170635.5937 - 48343.55068*j, 71222.43487 + 20178.39180*j;
    
    Dipole hD(x0,r);

    FieldCalculatorDipole fcd(hD, constants);

    Eigen::MatrixX3cd Ecode(3,3), Hcode(3,3);

    fcd.computeFields(Ecode, Hcode, xs);

    Eigen::Matrix3d E_difference_abs = (Ecode - unitTestsE).array().abs();

    std::cout << "Absolute difference" << std::endl;
    for (int i = 0; i < 3; ++i){
        std::cout << E_difference_abs.row(i) << std::endl;
    }



    return 0;
}