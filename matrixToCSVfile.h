#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <list>

using namespace std;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

// writing functions taking Eigen types as parameters, 
// see https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
template <typename Derived>

void matrixToCSVfile(string name, std::list<string> column_names, const Eigen::MatrixBase<Derived>& matrix)
{
    ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
    // file.close() is not necessary, 
    // desctructur closes file, see https://en.cppreference.com/w/cpp/io/basic_ofstream
}