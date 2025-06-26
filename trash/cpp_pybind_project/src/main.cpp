// #include <pybind11/embed.h>
// #include <iostream>

// namespace py = pybind11;

// // int main() {
// //     py::scoped_interpreter guard{}; // Start the Python interpreter

// //     try {
// //         py::module mod = py::module::import("modEigen");
// //         int result = mod.attr("add")(3, 4).cast<int>();
// //         std::cout << "Result of add(3, 4): " << result << std::endl;
// //     } catch (const std::exception &e) {
// //         std::cerr << "Python error: " << e.what() << std::endl;
// //         return 1;
// //     }

// //     return 0;
// // }

// #include <pybind11/embed.h> // everything needed for embedding
// namespace py = pybind11;

// int main() {
//     py::scoped_interpreter guard{}; // start the interpreter

//     try {
//         py::module_ my_script = py::module_::import("my_script");
//         int result = my_script.attr("multiply")(6, 7).cast<int>();
//         std::cout << "Result of multiply(6, 7): " << result << std::endl;
//     } catch (py::error_already_set &e) {
//         std::cerr << "Python error: " << e.what() << std::endl;
//     }

//     return 0;
// }

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
