#ifndef UTILS_PYBIND_H
#define UTILS_PYBIND_H

#include <Eigen/Dense>
#include <cmath>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/// @brief Namespace for utility functions related to pybind11 and Eigen matrices.
/// @details This namespace provides functions to convert between Eigen matrices and NumPy arrays,
/// as well as to call Python methods from C++ using pybind11.
namespace PybindUtils {

/// @brief Converts an Eigen matrix to a NumPy array.
/// @param mat The Eigen matrix to convert.
/// @return A pybind11 array_t<double> representing the Eigen matrix as a NumPy array.
/// @details The function uses row-major order for the conversion, which is suitable for Eigen matrices stored in row-major format.
inline py::array_t<double> eigen2numpy(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat) {
    return py::array_t<double>(
        {mat.rows(), mat.cols()},
        {sizeof(double) * mat.cols(), sizeof(double)},
        mat.data()
    );
}



// inline py::array_t<double> eigen2numpy(
//     const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat)
// {
//     return py::array_t<double>(
//         {mat.rows(), mat.cols()},
//         {sizeof(double) * mat.cols(), sizeof(double)},  // ✔ RowMajor stride: row jumps in blocks of cols
//         mat.data(),
//         py::capsule(mat.data(), [](void*) {/* no deallocation */})  // better than py::none()
//     );
// }

// inline py::array_t<double> eigen2numpy(
//     const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat) {
//     return py::array_t<double>(
//         {mat.rows(), mat.cols()},
//         {sizeof(double) * mat.cols(), sizeof(double)},
//         mat.data(),
//         py::none() // Do not let Python own the matrix 
//     );
// }

// inline py::array_t<double> eigen2numpy(Eigen::MatrixXd& mat) {
//     return py::array_t<double>(
//         {mat.rows(), mat.cols()},
//         {sizeof(double), sizeof(double) * mat.rows()},
//         mat.data(),
//         py::none()
//     );
// }

/// @brief Converts a NumPy array to an Eigen matrix.
/// @param nparray The NumPy array to convert.
/// @param rows The number of rows in the resulting Eigen matrix.
/// @param cols The number of columns in the resulting Eigen matrix.
/// @return An Eigen::MatrixXd representing the NumPy array as an Eigen matrix.
inline Eigen::Matrix<double, Eigen::Dynamic, 
                        Eigen::Dynamic, Eigen::RowMajor> numpy2eigen(py::array_t<double> nparray, int rows, int cols){
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    matrix((double*)nparray.request().ptr, rows, cols);
    // Eigen::Map<Eigen::MatrixXd> matrix((double*)nparray.request().ptr, rows, cols);
    return matrix;
}

/// @brief Calls the Python method `calculate_points` on a spline object.
/// @details This function retrieves points, normals, and tangents from a spline object using the specified resolution.
/// @param spline The Python object representing the spline.
/// @param resolution The resolution to use when calculating points.
/// @return A vector of Eigen::MatrixX3d containing points, normals, tangents1, and tangents2.
/// @throws std::runtime_error if the shapes of the NumPy arrays do not match the expected dimensions.
inline std::vector<Eigen::MatrixX3d> call_spline(py::object spline, int resolution) {

    py::tuple result = spline.attr("calculate_points")(resolution);

    const auto py_points = result[0].cast<py::array_t<double>>();
    const auto py_normals = result[1].cast<py::array_t<double>>();
    const auto py_tangents1 = result[2].cast<py::array_t<double>>();
    const auto py_tangents2 = result[3].cast<py::array_t<double>>();

    const int rows = py_points.shape(0), cols = py_points.shape(1);

    if (cols != 3) throw std::runtime_error("Expected shape (N, 3) for points");
    if (py_normals.shape(1) != 3) throw std::runtime_error("Expected shape (N, 3) for normals");

    // Convert
    const auto points = numpy2eigen(py_points, rows, cols);
    const auto normals = numpy2eigen(py_normals, rows, cols);
    const auto tangents1 = numpy2eigen(py_tangents1, rows, cols);
    const auto tangents2 = numpy2eigen(py_tangents2, rows, cols);

    return {points, normals, tangents1, tangents2};
}


// inline std::vector<Eigen::MatrixX3d> call_spline(py::object spline, int resolution) { 
//     py::tuple result = spline.attr("calculate_points")(resolution);
    
//     const auto py_points = result[0].cast<py::array_t<double>>();
//     const auto py_normals = result[1].cast<py::array_t<double>>();
//     const auto py_tangents1 = result[2].cast<py::array_t<double>>();
//     const auto py_tangents2 = result[3].cast<py::array_t<double>>();

//     const int rows = py_points.shape(0), cols = 3;

//     if (py_points.shape(1) != 3) throw std::runtime_error("Expected shape (N, 3) for points");
//     if (py_normals.shape(1) != 3) throw std::runtime_error("Expected shape (N, 3) for normals");

//     const auto points = numpy2eigen(py_points, rows, cols);
//     const auto normals = numpy2eigen(py_normals, rows, cols);
//     const auto tangents1 = numpy2eigen(py_tangents1, rows, cols);
//     const auto tangents2 = numpy2eigen(py_tangents2, rows, cols);

//     return {points, normals, tangents1, tangents2};
// }


// inline std::vector<Eigen::MatrixX3d> call_spline(py::object spline, int resolution){ 

//     std::cout << "call spline, l. 51" << std::endl;
//     std::cout << "Resolution: " << resolution << std::endl;
//     py::tuple result = spline.attr("calculate_points")(resolution);
    
//     std::cout << "call spline, l. 54" << std::endl;
//     py::array_t<double> py_points = result[0].cast<py::array_t<double>>();
//     py::array_t<double> py_normals = result[1].cast<py::array_t<double>>();
//     py::array_t<double> py_tangents1 = result[2].cast<py::array_t<double>>();
//     py::array_t<double> py_tangents2 = result[3].cast<py::array_t<double>>();

//     int rows = py_points.shape(0), cols = 3;

//     auto points = numpy2eigen(py_points, rows, cols);
//     auto normals = numpy2eigen(py_normals, rows, cols);
//     auto tangents1 = numpy2eigen(py_tangents1, rows, cols);
//     auto tangents2 = numpy2eigen(py_tangents2, rows, cols);

//     return {points, normals, tangents1, tangents2};
// }

}
#endif