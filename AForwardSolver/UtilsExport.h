#ifndef UTILS_EXPORT_H
#define UTILS_EXPORT_H

#include <Eigen/Dense>
#include <fstream>
#include <complex>
#include <string>
#include <sstream>

namespace Export {

inline std::string complexToString(const std::complex<double>& val) {
    std::ostringstream oss;
    oss << val.real() << "+" << val.imag() << "i";
    return oss.str();
}

inline void saveMatrixCSV(const std::string& filename, const Eigen::MatrixXcd& mat) {
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Failed to open file: " + filename);
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            file << complexToString(mat(i, j));
            if (j < mat.cols() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

inline void saveVectorCSV(const std::string& filename, const Eigen::VectorXcd& vec) {
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Failed to open file: " + filename);
    for (int i = 0; i < vec.size(); ++i) {
        file << complexToString(vec(i)) << "\n";
    }
    file.close();
}

inline void saveRealMatrixCSV(const std::string& filename, const Eigen::MatrixXd& mat) {
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Failed to open file: " + filename);
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            file << mat(i, j);
            if (j < mat.cols() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

inline void saveRealVectorCSV(const std::string& filename, const Eigen::VectorXd& vec) {
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Failed to open file: " + filename);
    for (int i = 0; i < vec.size(); ++i) {
        file << vec(i) << "\n";
    }
    file.close();
}

} // namespace Export

#endif // UTILS_EXPORT_H
