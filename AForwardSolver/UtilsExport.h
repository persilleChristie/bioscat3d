#ifndef UTILS_EXPORT_H
#define UTILS_EXPORT_H

#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>
#include <complex>
#include <list>

// Ensure output folder exists
inline void ensureOutputFolderExists(const std::string& folder = "FilesCSV") {
    std::filesystem::create_directory(folder);
}

// Save Eigen matrix (real or complex) with optional headers
template <typename Derived>
void matrixToCSVfile(const std::string& filename,
                     const std::list<std::string>& headers,
                     const Eigen::MatrixBase<Derived>& matrix) {
    ensureOutputFolderExists();
    std::ofstream file("FilesCSV/" + filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    // Write headers
    for (auto it = headers.begin(); it != headers.end(); ++it) {
        file << *it;
        if (std::next(it) != headers.end()) file << ",";
    }
    if (!headers.empty()) file << "\n";

    // Write matrix
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            file << matrix(i, j);
            if (j < matrix.cols() - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
}

// Save vector to CSV (real or complex)
template <typename Derived>
void saveVectorCSV(const std::string& filename, const Eigen::MatrixBase<Derived>& vector) {
    ensureOutputFolderExists();
    std::ofstream file("FilesCSV/" + filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < vector.size(); ++i) {
        file << vector(i) << "\n";
    }

    file.close();
}

// Save matrix to CSV without headers
template <typename Derived>
void saveMatrixCSV(const std::string& filename, const Eigen::MatrixBase<Derived>& matrix) {
    std::list<std::string> noHeaders;
    matrixToCSVfile(filename, noHeaders, matrix);
}

#endif // UTILS_EXPORT_H
