cmake_minimum_required(VERSION 3.14)
project(pybind_cpp_embed)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 REQUIRED)
set(Eigen3_DIR "/usr/share/eigen3/cmake")
find_package(Eigen3 REQUIRED)

# Build the C++ executable that embeds Python
add_executable(splinetest splinetest.cpp)
target_link_libraries(splinetest PRIVATE pybind11::embed)
target_link_libraries(splinetest PRIVATE Eigen3::Eigen)