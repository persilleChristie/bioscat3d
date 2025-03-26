#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include "spherePNT.h"

using namespace Eigen;
using namespace std;

constexpr double PI = 3.14159265358979323846;

void sphere(
    double radius,
    const Vector3d& center,
    int num_points,
    MatrixXd& points,
    MatrixXd& normals,
    MatrixXd& tau1,
    MatrixXd& tau2
) {
    int N = num_points;

    // Create theta and phi vectors (excluding first and last element as in Python)
    VectorXd theta0 = VectorXd::LinSpaced(N + 2, 0.0, PI).segment(1, N);
    VectorXd phi0 = VectorXd::LinSpaced(N + 2, 0.0, 2 * PI).segment(1, N);

    // Meshgrid equivalent
    MatrixXd theta(N, N), phi(N, N);
    for (int i = 0; i < N; ++i) {
        theta.row(i) = theta0.transpose();
        phi.col(i) = phi0;
    }

    ArrayXXd st = theta.array().sin();
    ArrayXXd ct = theta.array().cos();
    ArrayXXd sp = phi.array().sin();
    ArrayXXd cp = phi.array().cos();

    ArrayXXd x = center(0) + radius * st * cp;
    ArrayXXd y = center(1) + radius * st * sp;
    ArrayXXd z = center(2) + radius * ct;

    // Might be redundant
    int total_points = N * N;
    points.resize(total_points, 3);
    normals.resize(total_points, 3);
    tau1.resize(total_points, 3);
    tau2.resize(total_points, 3);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;

            Vector3d p(x(i, j), y(i, j), z(i, j));
            points.row(idx) = p;

            // Normal vector
            Vector3d n = (p - center).normalized();
            normals.row(idx) = n;

            // tau1 vector
            Vector3d t1(
                radius * ct(i, j) * cp(i, j),
                radius * ct(i, j) * sp(i, j),
                -radius * st(i, j)
            );
            tau1.row(idx) = t1.normalized();

            // tau2 vector
            Vector3d t2(
                -radius * st(i, j) * sp(i, j),
                radius * st(i, j) * cp(i, j),
                0.0
            );
            tau2.row(idx) = t2.normalized();
        }
    }
}

// Save points, normals, tau1, tau2 to CSV with headers and full precision
void saveSphereCSV(const std::string& filename,
    const MatrixXd& points,
    const MatrixXd& normals,
    const MatrixXd& tau1,
    const MatrixXd& tau2) {
std::ofstream outfile(filename);

if (outfile.is_open()) {
// Define full-precision CSV format
IOFormat csvFormat(FullPrecision, DontAlignCols, ",", "\n");

// Write header
outfile << "x,y,z,"
 << "nx,ny,nz,"
 << "t1x,t1y,t1z,"
 << "t2x,t2y,t2z\n";

// Combine all into one matrix (for formatting in one call)
MatrixXd full(points.rows(), 12);
full << points, normals, tau1, tau2;

// Write formatted output
outfile << full.format(csvFormat);
outfile.close();

std::cout << "Saved full sphere data with full precision to " << filename << "\n";
} else {
std::cerr << "Failed to write to " << filename << "\n";
}
}

int main() {
    Eigen::MatrixXd points, normals, tau1, tau2;
    Eigen::Vector3d center(0.0, 0.0, 0.0);
    double radius = 1.0;
    int num_points = 10;

    sphere(radius, center, num_points, points, normals, tau1, tau2);

    std::cout << "First point:\n" << points.row(0) << std::endl;
    std::cout << "First normal:\n" << normals.row(0) << std::endl;
    std::cout << "First tau1:\n" << tau1.row(0) << std::endl;
    std::cout << "First tau2:\n" << tau2.row(0) << std::endl;

    saveSphereCSV("GeneratedSphere.csv", points, normals, tau1, tau2);


    return 0;
}
