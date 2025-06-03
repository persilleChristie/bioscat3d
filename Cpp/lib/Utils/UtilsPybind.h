#ifndef UTILS_PYBIND_H
#define UTILS_PYBIND_H


#include <Eigen/Dense>
#include <cmath>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace PybindUtils {

/// @brief Translate Eigen matrix into numpy array without letting python own memory
/// @param mat Eigen matrix 
/// @return pybind11 array_t<double>
inline py::array_t<double> eigen2numpy(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat) {
    return py::array_t<double>(
        {mat.rows(), mat.cols()},
        {sizeof(double) * mat.cols(), sizeof(double)},
        mat.data(),
        py::none() // Do not let Python own the matrix 
    );
}
// inline py::array_t<double> eigen2numpy(Eigen::MatrixXd& mat) {
//     return py::array_t<double>(
//         {mat.rows(), mat.cols()},
//         {sizeof(double), sizeof(double) * mat.rows()},
//         mat.data(),
//         py::none()
//     );
// }

/// @brief Translate numpy array to row major Eigen matrix 
/// @param nparray py::array_t
/// @param rows rows of array
/// @param cols columns of array
/// @return Eigen matrix
inline Eigen::Matrix<double, Eigen::Dynamic, 
                        Eigen::Dynamic, Eigen::RowMajor> numpy2eigen(py::array_t<double> nparray, int rows, int cols){
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
    matrix((double*)nparray.request().ptr, rows, cols);
    // Eigen::Map<Eigen::MatrixXd> matrix((double*)nparray.request().ptr, rows, cols);

    return matrix;
}

inline std::vector<Eigen::MatrixX3d> call_spline(py::object spline, int resolution){ 

    std::cout << "call spline, l. 51" << std::endl;
    std::cout << "Resolution: " << resolution << std::endl;
    py::tuple result = spline.attr("calculate_points")(resolution);
    
    std::cout << "call spline, l. 54" << std::endl;
    py::array_t<double> py_points = result[0].cast<py::array_t<double>>();
    py::array_t<double> py_normals = result[1].cast<py::array_t<double>>();
    py::array_t<double> py_tangents1 = result[2].cast<py::array_t<double>>();
    py::array_t<double> py_tangents2 = result[3].cast<py::array_t<double>>();

    int rows = py_points.shape(0), cols = 3;

    auto points = numpy2eigen(py_points, rows, cols);
    auto normals = numpy2eigen(py_normals, rows, cols);
    auto tangents1 = numpy2eigen(py_tangents1, rows, cols);
    auto tangents2 = numpy2eigen(py_tangents2, rows, cols);

    return {points, normals, tangents1, tangents2};

}

}

#endif