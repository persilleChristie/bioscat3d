#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

constexpr double TOL = 1e-12;

bool parseLine(const std::string& line, std::vector<double>& values) {
    std::stringstream ss(line);
    std::string cell;
    values.clear();

    while (std::getline(ss, cell, ',')) {
        try {
            values.push_back(std::stod(cell));
        } catch (...) {
            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " file1.csv file2.csv\n";
        return 1;
    }

    std::ifstream file1(argv[1]), file2(argv[2]);
    if (!file1 || !file2) {
        std::cerr << "Error: could not open one of the files.\n";
        return 1;
    }

    std::string line1, line2;
    std::vector<double> values1, values2;
    int lineNumber = 0;

    // Skip headers
    std::getline(file1, line1);
    std::getline(file2, line2);

    auto getNextValidLine = [](std::ifstream& file, std::string& line) -> bool {
        while (std::getline(file, line)) {
            // Trim leading/trailing whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
    
            // Skip blank lines
            if (!line.empty()) return true;
        }
        return false;
    };
    
    while (true) {
        bool hasLine1 = getNextValidLine(file1, line1);
        bool hasLine2 = getNextValidLine(file2, line2);
    
        if (!hasLine1 && !hasLine2) break; // both EOF
        if (hasLine1 != hasLine2) {
            std::cerr << "Files have different lengths.\n";
            return 1;
        }
    
        ++lineNumber;
    
        if (!parseLine(line1, values1) || !parseLine(line2, values2)) {
            std::cerr << "Error parsing line " << lineNumber << ".\n";
            return 1;
        }
    
        if (values1.size() != values2.size()) {
            std::cerr << "Mismatch in number of values at line " << lineNumber << ".\n";
            return 1;
        }
    
        for (size_t i = 0; i < values1.size(); ++i) {
            if (std::abs(values1[i] - values2[i]) > TOL) {
                std::cerr << "Mismatch at line " << lineNumber 
                          << ", column " << (i + 1)
                          << ": " << values1[i] << " vs " << values2[i] << "\n";
                return 1;
            }
        }
    }
    
    if (file1.good() != file2.good()) {
        std::cerr << "Files have different lengths.\n";
        return 1;
    }

    std::cout << "Files are identical (within tolerance " << TOL << ").\n";
    return 0;
}
