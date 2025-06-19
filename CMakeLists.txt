# cmake_minimum_required(VERSION 3.14)
# project(bioscat3d)

# set(CMAKE_CXX_STANDARD 17)
# set(PYBIND11_FINDPYTHON ON)

# # Making sure packages work
# find_package(pybind11 REQUIRED)
# find_package(Eigen3 REQUIRED)
# #include_directories("/home/pernillechristie/rapidjson/include")
# include_directories(${CMAKE_SOURCE_DIR}/../rapidjson/include)


# # Copy Python script to the build dir so it can be found at runtime
# configure_file(${CMAKE_SOURCE_DIR}/Python/src/Spline.py ${CMAKE_BINARY_DIR}/Spline.py COPYONLY)

# # Allow us to compile all classes
# file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
#     ${CMAKE_SOURCE_DIR}/Cpp/src/Forward/*.cpp
#     ${CMAKE_SOURCE_DIR}/Cpp/src/Inverse/*.cpp
# )

# # Include directories for headers
# include_directories(${CMAKE_SOURCE_DIR}/Cpp/lib/Utils)
# include_directories(${CMAKE_SOURCE_DIR}/Cpp/lib/Forward)
# include_directories(${CMAKE_SOURCE_DIR}/Cpp/lib/Inverse)

# # add_executable(main ${SOURCES} main.cpp)
# # add_executable(main ${SOURCES} surfaceTester.cpp)
# add_executable(main ${SOURCES} pcnTester.cpp)


# target_link_libraries(main PRIVATE pybind11::embed Eigen3::Eigen)

# # Compile options (optional)
# target_compile_options(main PRIVATE -Wall -Wextra -O2)


cmake_minimum_required(VERSION 3.14)
project(bioscat3d)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Locate Python manually from Conda ---
execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
    OUTPUT_VARIABLE PYTHON_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Hardcode to shared library to avoid .a reference
set(PYTHON_LIB libpython3.11.so)

set(PYTHON_LIBRARY ${PYTHON_LIBRARY_DIR}/${PYTHON_LIB})

message(STATUS "Using Python include: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Using Python lib: ${PYTHON_LIBRARY}")

# --- Required packages ---
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED CONFIG)

# --- Include headers ---
include_directories(${CMAKE_SOURCE_DIR}/Cpp/lib/Utils)
include_directories(${CMAKE_SOURCE_DIR}/Cpp/lib/Forward)
include_directories(${CMAKE_SOURCE_DIR}/Cpp/lib/Inverse)
include_directories(${CMAKE_SOURCE_DIR}/../rapidjson/include)
#include_directories("/home/pernillechristie/rapidjson/include")
include_directories(${PYTHON_INCLUDE_DIR})

# --- Copy Python file to build dir ---
configure_file(${CMAKE_SOURCE_DIR}/Python/src/Spline.py ${CMAKE_BINARY_DIR}/Spline.py COPYONLY)

# --- Collect source files ---
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/Cpp/src/Forward/*.cpp
    ${CMAKE_SOURCE_DIR}/Cpp/src/Inverse/*.cpp
)

# --- Build target ---
add_executable(main ${SOURCES} Cpp/src/Validation/Tangential_error.cpp)
# add_executable(main ${SOURCES} surfaceTester.cpp)
# add_executable(main ${SOURCES} pcnTester.cpp)

target_link_libraries(main PRIVATE Eigen3::Eigen ${PYTHON_LIBRARY})
# Apply proper flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(main PRIVATE -Wall -Wextra -g -O0)
else()
    target_compile_options(main PRIVATE -Wall -Wextra -O2)
endif()

# # cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ..

# cmake_minimum_required(VERSION 3.14)
# project(bioscat3d)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# # --- Python + NumPy + pybind11 ---
# find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
# find_package(pybind11 REQUIRED CONFIG)
# find_package(Eigen3 REQUIRED)
# find_package(Threads REQUIRED)

# # --- Include Directories ---
# include_directories(${CMAKE_SOURCE_DIR}/Cpp/lib/Utils)
# include_directories(${CMAKE_SOURCE_DIR}/Cpp/lib/Forward)
# include_directories(${CMAKE_SOURCE_DIR}/Cpp/lib/Inverse)
# include_directories("/home/pernillechristie/rapidjson/include")
# include_directories(${Python3_INCLUDE_DIRS})
# include_directories(${Python3_NumPy_INCLUDE_DIRS})

# # --- Sources ---
# file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
#     ${CMAKE_SOURCE_DIR}/Cpp/src/Forward/*.cpp
#     ${CMAKE_SOURCE_DIR}/Cpp/src/Inverse/*.cpp
# )

# # --- Executable ---
# add_executable(main ${SOURCES} realDataArray.cpp)
# # add_executable(main ${SOURCES} surfaceTester.cpp)

# target_link_libraries(main
#     PRIVATE Eigen3::Eigen
#             pybind11::embed
#             Python3::Python
#             Threads::Threads
# )

# # --- Debug Flags ---
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(main PRIVATE -Wall -Wextra -g -O0)
# else()
#     target_compile_options(main PRIVATE -Wall -Wextra -O2)
# endif()