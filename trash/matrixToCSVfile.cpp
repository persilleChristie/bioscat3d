#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <list>

#include "matrixToCSVfile.h"

using namespace std;

// code from https://stackoverflow.com/a/66560639 

// writing functions taking Eigen types as parameters, 
// see https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
template <typename Derived>
void matrixToCSVfile(string name, std::list<string> column_names, const Eigen::MatrixBase<Derived>& matrix) // 
{
    ofstream file(name);
    /*
    */

    // Open file
    // std::ofstream file("matrix.csv");
    
    if (file.is_open()) {
        for (auto name : column_names){
            file << name << ",";
        }
        file << "\n";

        // Write matrix to file
        file << matrix.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", "\n"));
        file.close();
        std::cout << "Matrix saved to file\n";
    } else {
        std::cerr << "Unable to open file.\n";
    }
}