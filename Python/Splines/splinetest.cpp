#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <iostream>

namespace py = pybind11;

// Helper function to create meshgrid (like numpy.meshgrid)
void create_meshgrid(const Eigen::VectorXd& x, const Eigen::VectorXd& y,
                     py::array_t<double>& Xout, py::array_t<double>& Yout) {
    ssize_t nx = x.size();
    ssize_t ny = y.size();

    // Allocate output arrays (shape: [ny, nx], like NumPy)
    Xout = py::array_t<double>({ny, nx});
    Yout = py::array_t<double>({ny, nx});

    auto Xbuf = Xout.mutable_unchecked<2>();
    auto Ybuf = Yout.mutable_unchecked<2>();

    for (ssize_t i = 0; i < ny; ++i) {
        for (ssize_t j = 0; j < nx; ++j) {
            Xbuf(i, j) = x(j);
            Ybuf(i, j) = y(i);
        }
    }
}

int main() {
    // 1. Start Python interpreter
    py::scoped_interpreter guard{};

    // 2. Add the path to the Python file if needed
    py::module_::import("sys").attr("path").attr("insert")(1, "./");

    // 3. Import your Python module
    py::module_ spline = py::module_::import("Spline");

    // 4. Prepare data in Eigen and convert to NumPy arrays
    // Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 100);
    // Eigen::MatrixXd Y = Eigen::MatrixXd::Random(100, 100);

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(100, -1.0, 1.0);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(100, -1.0, 1.0);

    py::array_t<double> Xfine, Yfine;
    create_meshgrid(x, y, Xfine, Yfine);
    Eigen::MatrixXd Z = Eigen::MatrixXd::Random(100, 100);

    // auto Xfine = py::array_t<double>({X.rows(), X.cols()}, X.data());
    // auto Yfine = py::array_t<double>({Y.rows(), Y.cols()}, Y.data());
    auto Zfine = py::array_t<double>({Z.rows(), Z.cols()}, Z.data());

    // 5. Initialize spline
    spline.attr("SplineManager").attr("initialize")(Xfine, Yfine, Zfine);

    // 6. Call calculate
    int resolution = 100;
    py::tuple result = spline.attr("SplineManager").attr("calculate")(resolution).cast<py::tuple>();

    std::cout << "Got " << result.size() << " arrays from spline" << std::endl;

    return 0;
}
