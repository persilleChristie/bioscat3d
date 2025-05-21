cmake_minimum_required(VERSION 3.14)
project(pybind_cpp_embed)

set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 REQUIRED)

# Build the Python module
pybind11_add_module(modEigen src/modEigen.cpp)

# Build the C++ executable that embeds Python
add_executable(run_pybind src/main.cpp)
target_link_libraries(run_pybind PRIVATE pybind11::embed)
