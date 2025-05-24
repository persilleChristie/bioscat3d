cmake_minimum_required(VERSION 3.14)
project(bioscat3d)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_FINDPYTHON ON)

# Making sure packages work
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
#include_directories("/home/pernillechristie/rapidjson/include")
include_directories(${CMAKE_SOURCE_DIR}/../rapidjson/include)


# Copy Python script to the build dir so it can be found at runtime
configure_file(${CMAKE_SOURCE_DIR}/Python/src/Spline.py ${CMAKE_BINARY_DIR}/Spline.py COPYONLY)

# Allow us to compile all classes
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/Cpp/src/Forward/*.cpp
)

# Include directories for headers
include_directories(${CMAKE_SOURCE_DIR}/Cpp/lib/Utils)
include_directories(${CMAKE_SOURCE_DIR}/Cpp/lib/Forward)

add_executable(main ${SOURCES} main.cpp)

target_link_libraries(main PRIVATE pybind11::embed Eigen3::Eigen)

# Compile options (optional)
target_compile_options(main PRIVATE -Wall -Wextra -O2)