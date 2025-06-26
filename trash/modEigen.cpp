#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>

namespace py = pybind11;

void modify_matrix(Eigen::MatrixXd& mat) {
    py::scoped_interpreter guard{};
    
    py::object pyfunc = py::eval(R"(
def f(M):
    M[0, 0] += 1000
    M[:, 1] *= 2
    return M
f
)");

    pyfunc(mat);  // modifies in-place
}

int main() {
    Eigen::MatrixXd mat(3, 3);
    mat << 1, 2, 3,
           4, 5, 6,
           7, 8, 9;

    std::cout << "Before:\n" << mat << "\n";

    modify_matrix(mat);

    std::cout << "After:\n" << mat << "\n";

    return 0;
}
