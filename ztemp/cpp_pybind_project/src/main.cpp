#include <pybind11/embed.h>
#include <iostream>

// namespace py = pybind11;

// int main() {
//     py::scoped_interpreter guard{}; // Start the Python interpreter

//     try {
//         py::module mod = py::module::import("modEigen");
//         int result = mod.attr("add")(3, 4).cast<int>();
//         std::cout << "Result of add(3, 4): " << result << std::endl;
//     } catch (const std::exception &e) {
//         std::cerr << "Python error: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }

#include <pybind11/embed.h> // everything needed for embedding
namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{}; // start the interpreter

    try {
        py::module_ my_script = py::module_::import("my_script");
        int result = my_script.attr("multiply")(6, 7).cast<int>();
        std::cout << "Result of multiply(6, 7): " << result << std::endl;
    } catch (py::error_already_set &e) {
        std::cerr << "Python error: " << e.what() << std::endl;
    }

    return 0;
}
