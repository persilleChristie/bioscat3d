#include <iostream>
#include <fstream>
#include <Eigen/Dense>

int main() {
    // Create a 3x3 matrix
    Eigen::MatrixXd mat(3, 3);
    mat << 1.1, 2.2, 3.3,
           4.4, 5.5, 6.6,
           7.7, 8.8, 9.9;

    // Open file
    std::ofstream file("matrix.csv");
    
    if (file.is_open()) {
        // Write matrix to file
        file << mat.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", "\n"));
        file.close();
        std::cout << "Matrix saved to matrix.csv\n";
    } else {
        std::cerr << "Unable to open file.\n";
    }

    return 0;
}
