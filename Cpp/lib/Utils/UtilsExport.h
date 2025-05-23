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

inline void saveSurfaceDataCSV(
    const std::string& filename,
    const Eigen::MatrixXd& x,
    const Eigen::MatrixXd& tau1,
    const Eigen::MatrixXd& tau2,
    const Eigen::MatrixXd& normal)
{
    if (x.cols() != 3 || tau1.cols() != 3 || tau2.cols() != 3 || normal.cols() != 3)
        throw std::invalid_argument("All matrices must have 3 columns.");
    if (x.rows() != tau1.rows() || x.rows() != tau2.rows() || x.rows() != normal.rows())
        throw std::invalid_argument("All matrices must have the same number of rows.");

    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Failed to open file: " + filename);

    file << "x,y,z,tau1_x,tau1_y,tau1_z,tau2_x,tau2_y,tau2_z,normal_x,normal_y,normal_z\n";

    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < 3; ++j) file << x(i, j) << (j < 2 ? "," : ",");
        for (int j = 0; j < 3; ++j) file << tau1(i, j) << (j < 2 ? "," : ",");
        for (int j = 0; j < 3; ++j) file << tau2(i, j) << (j < 2 ? "," : ",");
        for (int j = 0; j < 3; ++j) file << normal(i, j) << (j < 2 ? "," : "");
        file << "\n";
    }

    file.close();
}


} // namespace Export

#endif // UTILS_EXPORT_H
