#include <gtest/gtest.h>
#include <Eigen/Dense>

// Example test
TEST(EigenTest, VectorAddition) {
    Eigen::Vector3d a(1, 2, 3);
    Eigen::Vector3d b(4, 5, 6);
    Eigen::Vector3d c = a + b;
    EXPECT_EQ(c(0), 5);
    EXPECT_EQ(c(1), 7);
    EXPECT_EQ(c(2), 9);
}