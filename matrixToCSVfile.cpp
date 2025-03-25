#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <list>

#include "matrixToCSVfile.h"

using namespace std;

// code from https://stackoverflow.com/a/66560639 

// define the format you want, you only need one instance of this...
// see https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

// writing functions taking Eigen types as parameters, 
// see https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
template <typename Derived>
void matrixToCSVfile(string name, std::list<string> column_names, const Eigen::MatrixBase<Derived>& matrix)
{
    ofstream file(name.c_str());
    for (auto name : column_names){
        file << name << ",";
    }
    file << '\n';
    file << matrix.format(CSVFormat);
    // file.close() is not necessary, 
    // desctructur closes file, see https://en.cppreference.com/w/cpp/io/basic_ofstream
}