
# Making sure packages work
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(runUnitTests test_utils.cpp)
target_link_libraries(runUnitTests ${GTEST_LIBRARIES} pthread)
add_test(NAME runUnitTests COMMAND runUnitTests)