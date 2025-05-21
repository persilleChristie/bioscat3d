#include <pybind11/embed.h>
#include <iostream>

namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{};
    std::cout << "Python embedded successfully!" << std::endl;
    return 0;
}
