#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{}; // Start the Python interpreter

    std::vector<double> data = {1.0, 2.0, 3.0};

    std::cout << "Before Python call: ";
    for (double d : data) std::cout << d << " ";
    std::cout << std::endl;

    // Wrap vector memory in a NumPy array (no copy)
    py::array_t<double> array(
        {static_cast<py::ssize_t>(data.size())}, // shape
        {sizeof(double)},                        // stride
        data.data(),                             // pointer to data
        py::cast(nullptr)                        // no ownership (C++ owns memory)
    );

    // Import your Python script (must be in PYTHONPATH)
    try {
        py::module py_mod = py::module::import("modify_array");
        py_mod.attr("modify")(array);  // call function modify(np_array)

        std::cout << "After Python call: ";
        for (double d : data) std::cout << d << " ";
        std::cout << std::endl;

    } catch (py::error_already_set &e) {
        std::cerr << "Python error:\n" << e.what() << std::endl;
    }

    return 0;
}
