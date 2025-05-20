// py::module spline = py::module::import("spline_wrapper");

// py::array Xfine = ...; // pass in from C++/Eigen
// py::array Yfine = ...;
// py::array Zfine = ...;
// spline.attr("SplineManager").attr("initialize")(Xfine, Yfine, Zfine);

// // Now just call the method multiple times without reinitializing
// int resolution = 100;
// auto result = spline.attr("SplineManager").attr("calculate")(resolution);

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <iostream>

namespace py = pybind11;

int main() {
    // 1. Start Python interpreter
    py::scoped_interpreter guard{};

    // 2. Add the path to the Python file if needed
    py::module::import("sys").attr("path").attr("insert")(1, "./");

    // 3. Import your Python module
    py::module spline = py::module::import("spline_wrapper");

    // 4. Prepare data in Eigen and convert to NumPy arrays
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 100);
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(100, 100);
    Eigen::MatrixXd Z = Eigen::MatrixXd::Random(100, 100);

    auto Xfine = py::array_t<double>({X.rows(), X.cols()}, X.data());
    auto Yfine = py::array_t<double>({Y.rows(), Y.cols()}, Y.data());
    auto Zfine = py::array_t<double>({Z.rows(), Z.cols()}, Z.data());

    // 5. Initialize spline
    spline.attr("SplineManager").attr("initialize")(Xfine, Yfine, Zfine);

    // 6. Call calculate
    int resolution = 100;
    py::tuple result = spline.attr("SplineManager").attr("calculate")(resolution).cast<py::tuple>();

    std::cout << "Got " << result.size() << " arrays from spline" << std::endl;

    return 0;
}
