#ifndef UTILS_EXPORT_H
#define UTILS_EXPORT_H

#include <Eigen/Dense>
#include <fstream>
#include <complex>
#include <string>
#include <sstream>

namespace Export {

/// @brief Converts a complex number to a string representation.
/// @param val The complex number to convert.
/// @return A string representation of the complex number in the format "real+imagi".
/// @details The function formats the complex number as "real+imagi", where "real" is the real part and "imagi" is the imaginary part.
inline std::string complexToString(const std::complex<double>& val) {
    std::ostringstream oss;
    oss << val.real() << "+" << val.imag() << "i";
    return oss.str();
}

/// @brief Converts a complex number to a string representation.
/// @details Formats as "real+imagi" using default stream precision.
/// @param val The complex number to convert.
/// @param precision The number of decimal places to include in the string representation.
/// @return A string representation of the complex number in the format "real+imagi", with the specified precision.
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
/// @brief Saves a complex vector to a CSV file.
/// @param filename The name of the file to save the vector to.
/// @param vec The complex vector to save.
/// @details The function opens the specified file and writes each element of the vector to a new line in the format "real+imagi".
inline void saveVectorCSV(const std::string& filename, const Eigen::VectorXcd& vec) {
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Failed to open file: " + filename);
    for (int i = 0; i < vec.size(); ++i) {
        file << complexToString(vec(i)) << "\n";
    }
    file.close();
}

/// @brief Saves a real matrix to a CSV file.
/// @param filename The name of the file to save the matrix to.
/// @param mat The real matrix to save.
/// @details The function opens the specified file and writes each element of the matrix in CSV format, with each row on a new line.
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

/// @brief Saves a real vector to a CSV file.
/// @param filename The name of the file to save the vector to.
/// @param vec The real vector to save.
/// @details The function opens the specified file and writes each element of the vector to a new line.
inline void saveRealVectorCSV(const std::string& filename, const Eigen::VectorXd& vec) {
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Failed to open file: " + filename);
    for (int i = 0; i < vec.size(); ++i) {
        file << vec(i) << "\n";
    }
    file.close();
}

/// @brief Saves surface data to a CSV file.
/// @param filename The name of the file to save the surface data to.
/// @param x The matrix of points on the surface (Nx3).
/// @param tau1 The first tangent vector at the surface points (Nx3).
/// @param tau2 The second tangent vector at the surface points (Nx3).
/// @param normal The normal vector at the surface points (Nx3).
/// @details The function checks that all matrices have 3 columns and the same number of rows,
/// and then writes the data to the specified CSV file in a structured format.
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
