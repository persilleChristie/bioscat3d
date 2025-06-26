#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{};
    try {
        std::vector<double> data = {1.0, 2.0, 3.0};

        std::cout << "Data before Python call: ";
        for (const auto &val : data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        // Create numpy array wrapper around C++ vector (no copy)
        py::array_t<double> arr(
            {static_cast<py::ssize_t>(data.size())},
            {sizeof(double)},
            data.data(),
            py::none() // no ownership, do NOT let Python delete it
        );

        // Import add.py and call the add function
        py::module add_mod = py::module::import("add");
        py::function add_func = add_mod.attr("add");

        add_func(arr);  // modify the data in-place

        std::cout << "Data after Python call: ";
        for (const auto &val : data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

    } catch (py::error_already_set &e) {
        std::cerr << "Python error:\n" << e.what() << std::endl;
    }
    return 0;
}
