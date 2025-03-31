#include <iostream>
#include <Eigen/Dense>
#include <iostream>
#include <complex>
#include <fstream>

using namespace Eigen;
using namespace std;

const complex<double> j(0, 1);  // Imaginary unit

const double omega = 1.0;
const double epsilon0 = 1.0; //constants.epsilon0;
const double n0 = 1.0;
const double mu0 = 1.0; //constants.mu0;
const double k0 = 1.0;// constants.k0;

const double epsilon_air = epsilon0;
const double eta_air = 1.0; //constants.eta0; //sqrt(omega/epsilon0);

const double pi = 3.14159265358979323846264338327950288;

// =========================================
//  Compute Angular Components (Pass by Reference)
// =========================================
void computeAngles(const Eigen::Vector3d& x, double r, 
    double& cosTheta, double& sinTheta, 
    double& cosPhi, double& sinPhi) {
if (r == 0) {
cosTheta = sinTheta = 0.0;
cosPhi = 1.0; sinPhi = 0.0;
return;
}

cosTheta = x[2] / r;
sinTheta = sqrt(1 - cosTheta * cosTheta);
double xy_norm = std::hypot(x[0], x[1]);

if (xy_norm == 0) {
cosPhi = 1.0;
sinPhi = 0.0;
} else {
cosPhi = x[0] / xy_norm;
sinPhi = x[1] / xy_norm;
}
}

// =========================================
//  Compute E-Field Components (Spherical, Precompute E_r & E_theta)
// =========================================
void compute_E_fields(double r, double Iel, double cos_theta, double sin_theta, 
const complex<double>& expK0r, complex<double>& E_r_out, 
complex<double>& E_theta_out) {
if (r == 0) {
E_r_out = E_theta_out = 0.0;
return;
}

E_r_out = eta_air * Iel * cos_theta / (2.0 * pi * r * r) 
* (1.0 + 1.0 / (j * k0 * r)) * expK0r;

E_theta_out = (j * eta_air * Iel * sin_theta / (4.0 * pi * r)) 
* (1.0 + 1.0 / (j * k0 * r) - 1.0 / (k0 * r * r)) * expK0r;
}

// =========================================
//  Compute E-Field Components (Cartesian, No Redundant Calls)
// =========================================
Eigen::Vector3cd compute_E_cartesian(double sin_theta, double cos_theta, 
              double sin_phi, double cos_phi, 
              const complex<double>& E_r, 
              const complex<double>& E_theta) {
return {
E_r * sin_theta * cos_phi + E_theta * cos_theta * cos_phi,
E_r * sin_theta * sin_phi + E_theta * cos_theta * sin_phi,
E_r * cos_theta - E_theta * sin_theta
};
}

// =========================================
//  Compute H-Field Components (Precompute H_phi)
// =========================================
void compute_H_fields(double r, double Iel, double sin_theta, 
const complex<double>& expK0r, complex<double>& H_phi_out) {
if (r == 0) {
H_phi_out = 0.0;
return;
}

H_phi_out = j * k0 * Iel * sin_theta / (4.0 * pi * r) 
* (1.0 + 1.0 / (j * k0 * r)) * expK0r;
}

// =========================================
//  Compute Both E and H Fields (Precompute Values)
// =========================================
pair<Eigen::Vector3cd, Eigen::Vector3cd> fieldDipole(double r, double Iel, 
double cos_theta, double sin_theta, double cos_phi, double sin_phi, 
const complex<double>& expK0r) {

// Precompute E_r and E_theta
complex<double> E_r_val, E_theta_val;
compute_E_fields(r, Iel, cos_theta, sin_theta, expK0r, E_r_val, E_theta_val);

// Compute E-field in Cartesian coordinates
Eigen::Vector3cd E = compute_E_cartesian(sin_theta, cos_theta, sin_phi, cos_phi, E_r_val, E_theta_val);

// Precompute H_phi
complex<double> H_phi_val;
compute_H_fields(r, Iel, sin_theta, expK0r, H_phi_val);

// Compute H-field
Eigen::Vector3cd H = {
-H_phi_val * sin_phi,
H_phi_val * cos_phi,
0.0
};

return {E, H};
}


inline double norm(const Eigen::Vector3d& x) {
return std::hypot(x[0], x[1], x[2]);
}

inline double cos_theta(const Eigen::Vector3d& rPrime) {
return rPrime[2] / norm(rPrime);
}

inline double sin_theta(const Eigen::Vector3d& rPrime) {
double cosT = cos_theta(rPrime);
cosT = std::clamp(cosT, -1.0, 1.0);
return sqrt(1 - cosT * cosT);
}

/*
cos_phi = np.sign(rPrime[1]) * rPrime[0] / np.sqrt(rPrime[0]**2 + rPrime[1]**2)
sin_phi = np.sqrt(1 - cos_phi**2)
*/

inline double cos_phi(const Eigen::Vector3d& rPrime) {
double xy_norm = std::hypot(rPrime[0], rPrime[1]);
return (xy_norm == 0) ? 1.0 : rPrime[0] / xy_norm;
}

inline double sin_phi(const Eigen::Vector3d& rPrime) {
double xy_norm = std::hypot(rPrime[0], rPrime[1]);
return (xy_norm == 0) ? 0.0 : rPrime[1] / xy_norm;
}


// =========================================
//  Compute Rotation Matrices
// =========================================
/*
# rotation matrix for theta around y axis
Ry = np.array([[cos_theta, 0, sin_theta],
[0, 1, 0],
[-sin_theta, 0, cos_theta]])
*/

Eigen::Matrix3d rotation_matrix_y(const Eigen::Vector3d& rPrime) {
double cosT = cos_theta(rPrime);
double sinT = sin_theta(rPrime);
Eigen::Matrix3d Ry {{ cosT,  0, sinT },
 { 0,     1,   0  },
 { -sinT, 0, cosT }
 };
return Ry;
}

/*
# rotation matrix for phi around z axis
Rz = np.array([[cos_phi, -sin_phi, 0],
[sin_phi, cos_phi, 0],
[0, 0, 1]])
*/

Eigen::Matrix3d rotation_matrix_z(const Eigen::Vector3d& rPrime) {
double cosP = cos_phi(rPrime);
double sinP = sin_phi(rPrime);
Eigen::Matrix3d Rz {{cosP, -sinP, 0} ,
 {sinP,  cosP, 0} ,
 {0,      0,   1}
 };
return Rz;
}


Eigen::Matrix3d rotation_matrix_y_inv(const Eigen::Vector3d& rPrime) {
double cosT = cos_theta(rPrime);
double sinT = sin_theta(rPrime);
Eigen::Matrix3d Ry {{ cosT,  0, -sinT },
 { 0,     1,   0  },
 { sinT,  0,  cosT }
 };
return Ry;
}


Eigen::Matrix3d rotation_matrix_z_inv(const Eigen::Vector3d& rPrime) {
double cosP = cos_phi(rPrime);
double sinP = sin_phi(rPrime);
Eigen::Matrix3d Rz {{cosP,  sinP, 0} ,
 {-sinP, cosP, 0} ,
 {0,      0,   1}
 };
return Rz;
}


// =========================================
//  Function to Move the Dipole Field
// =========================================

pair<Eigen::Vector3cd, Eigen::Vector3cd> moveDipoleField(
const Eigen::Vector3d& xPrime,
const Eigen::Vector3d& rPrime,
const Eigen::Vector3cd& E,
const Eigen::Vector3cd& H
) {
// Compute rotation matrices
auto Ry_inv = rotation_matrix_y_inv(rPrime);
auto Rz_inv = rotation_matrix_z_inv(rPrime);

// Rotate the field components
Vector3cd E_rot = Ry_inv * Rz_inv * E;
Vector3cd H_rot = Ry_inv * Rz_inv * H;

// Translate the fields
Vector3cd E_prime = E_rot - xPrime;
Vector3cd H_prime = H_rot - xPrime;

return {E_prime, H_prime};
}



pair<Eigen::Vector3cd, Eigen::Vector3cd> hertzianDipoleField(const Eigen::Vector3d& x,      // point to evaluate field in
const double Iel,               // "size" of Hertzian dipole
const Eigen::Vector3d& xPrime,  // Hertzian dipole placement
const Eigen::Vector3d& rPrime  // Hertzian dipole direction
)
{
auto Ry = rotation_matrix_y(rPrime);
auto Rz = rotation_matrix_z(rPrime);

Eigen::Vector3d x_origo = Rz * Ry * (x - xPrime);
double r = x_origo.norm();
double cos_theta = 0.0, sin_theta = 0.0, cos_phi = 0.0, sin_phi = 0.0;

computeAngles(x_origo, r, cos_theta, sin_theta, cos_phi, sin_phi);

// Compute exp(-j * k0 * r) once
complex<double> expK0r = polar(1.0, -k0 * r);

auto [E_origo, H_origo] = fieldDipole(r, Iel, cos_theta, sin_theta, cos_phi, sin_phi, expK0r);

auto [E, H] = moveDipoleField(xPrime, rPrime, E_origo, H_origo);

return {E, H};
}

pair<Eigen::MatrixX3cd, Eigen::MatrixX3cd> evaluateAtPoints(const double Iel,               // "size" of Hertzian dipole
                                                            const Eigen::Vector3d& xPrime,  // Hertzian dipole placement
                                                            const Eigen::Vector3d& rPrime,  // Hertzian dipole direction
                                                            const MatrixX3d testpoints)
{

    int N = testpoints.rows();


    Eigen::MatrixX3cd E(N, 3);
    Eigen::MatrixX3cd H(N, 3);
    
    for (int i = 0; i < N; ++i){
        Vector3d testpoint = testpoints.row(i);

        auto [E_row, H_row] = hertzianDipoleField(testpoint, Iel, xPrime, rPrime);

        E.row(i) << E_row[0], E_row[1], E_row[2];
        H.row(i) << H_row[0], H_row[1], H_row[2];
    }

    return {E, H};
}

void saveToCSV(const MatrixXcd& E, const MatrixXcd& H, const string& filename) {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return;
    }

    // Write header
    file << "Ex_Re,Ex_Im,Ey_Re,Ey_Im,Ez_Re,Ez_Im,"
         << "Hx_Re,Hx_Im,Hy_Re,Hy_Im,Hz_Re,Hz_Im\n";

    // Write data
    for (int i = 0; i < E.rows(); ++i) {
        file << E(i, 0).real() << "," << E(i, 0).imag() << ","
             << E(i, 1).real() << "," << E(i, 1).imag() << ","
             << E(i, 2).real() << "," << E(i, 2).imag() << ","
             << H(i, 0).real() << "," << H(i, 0).imag() << ","
             << H(i, 1).real() << "," << H(i, 1).imag() << ","
             << H(i, 2).real() << "," << H(i, 2).imag() << "\n";
    }

    file.close();
}


struct Parameter {
    std::string name;
    double value;
};

int main(int argc, char* argv[]){
    // Subtracting file names (first argument is path to program)
    string parameter_file = argv[1];
    string testpoint_file = argv[2];
    string output_file = argv[3];

    // Open the CSV file
    std::ifstream file(parameter_file); // Replace with your CSV filename
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }
    
    std::vector<Parameter> params;
    std::string line;
    
    // Read the first line (column titles) and ignore it
    std::getline(file, line);
    
    // Read the remaining CSV data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string name, value_str;
        if (std::getline(ss, name, ',') && std::getline(ss, value_str, ',')) {
            try {
                double value = std::stod(value_str);
                params.push_back({name, value});
            } catch (const std::exception &e) {
                std::cerr << "Error parsing value for " << name << "\n";
            }
        }
    }
    file.close();
    
    // Initialize Eigen vectors
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    
    // Assign values to Eigen vectors
    for (const auto& param : params) {
        if (param.name == "position_x") position(0) = param.value;
        else if (param.name == "position_y") position(1) = param.value;
        else if (param.name == "position_z") position(2) = param.value;
        else if (param.name == "direction_x") direction(0) = param.value;
        else if (param.name == "direction_y") direction(1) = param.value;
        else if (param.name == "direction_z") direction(2) = param.value;
    }
    
    // Read test points into an Eigen matrix
    std::ifstream matrix_file(testpoint_file); // Replace with your test points CSV filename
    if (!matrix_file.is_open()) {
        std::cerr << "Error opening test points file!" << std::endl;
        return 1;
    }
    
    std::getline(matrix_file, line); // Ignore the first line (column titles)
    std::vector<Eigen::Vector3d> testpoints_vec;
    
    while (std::getline(matrix_file, line)) {
        std::stringstream ss(line);
        std::string value_str;
        double values[3] = {0.0, 0.0, 0.0};
        int col = 0;
        
        while (std::getline(ss, value_str, ',') && col < 3) {
            try {
                values[col] = std::stod(value_str);
            } catch (const std::exception &e) {
                std::cerr << "Error parsing test point value at column " << col << "\n";
                values[col] = 0.0; // Default to zero if parsing fails
            }
            col++;
        }
        
        if (col == 3) { // Ensure full row of 3 values
            testpoints_vec.push_back(Eigen::Vector3d(values[0], values[1], values[2]));
        }
    }
    matrix_file.close();

    int test_vec_size = static_cast<int>(testpoints_vec.size());

    // Convert vector to Eigen matrix
    Eigen::MatrixXd testpoints(test_vec_size, 3);
    for (int i = 0; i < test_vec_size; ++i) {
        testpoints.row(i) = testpoints_vec[i].transpose();
    };

    double Iel = 1;

    auto [E, H] = evaluateAtPoints(Iel, position, direction, testpoints);

    ////////////////////////////////
    // SAVE TO CSV
    saveToCSV(E, H, output_file);

    return 0;
}