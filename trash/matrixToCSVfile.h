#ifndef _MATRIX_TO_CSV_H
#define _MATRIX_TO_CSV_H

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <list>

using namespace std;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

template <typename Derived>
void matrixToCSVfile(string name, std::list<string> column_names, const Eigen::MatrixBase<Derived>& matrix); // 

#endif