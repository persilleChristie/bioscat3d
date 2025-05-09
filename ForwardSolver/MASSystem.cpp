#include <fstream> 
#include "MASSystem.h"
#include "Constants.h"
#include "rapidjson/document.h" 
#include "rapidjson/istreamwrapper.h"


MASSystem::MASSystem(const char* jsonPath, const std::string surfaceType)//, Constants& constants)
    // : constants_(constants)
    {
        if (surfaceType == "Bump"){
            generateBumpSurface(jsonPath);
        } else if (surfaceType == "GP"){
            generateGPSurface(jsonPath);
        } else {
            std::cerr << "[Error] Surface type not recognised! Allowed types are 'Bump' and 'GP'\n";
        }

    }


/**
    * Approximate second derivative in x using central differences (to mimick numpy's gradient)
    * @param Z: Matrix with z values in grid
    * @param x: Vector of x values
    * 
    * @return Matrix of the second derivative
    */
   Eigen::MatrixXd second_derivative_x(const Eigen::MatrixXd& Z, const Eigen::VectorXd& x) {
    int rows = Z.rows();
    int cols = Z.cols();
    Eigen::MatrixXd d2Z_dx2 = Eigen::MatrixXd::Zero(rows, cols);

    for (int j = 1; j < cols - 1; ++j) {
        double dx_prev = x(j) - x(j - 1);
        double dx_next = x(j + 1) - x(j);
        for (int i = 0; i < rows; ++i) {
            double z_prev = Z(i, j - 1);
            double z_curr = Z(i, j);
            double z_next = Z(i, j + 1);
            double dx = 0.5 * (dx_prev + dx_next);
            d2Z_dx2(i, j) = (z_next - 2 * z_curr + z_prev) / (dx * dx);
        }
    }

    return d2Z_dx2;
}

/**
    * Approximate second derivative in y using central differences (to mimick numpy's gradient)
    * @param Z: Matrix with z values in grid
    * @param y: Vector of y values
    * 
    * @return Matrix of the second derivative
    */
Eigen::MatrixXd second_derivative_y(const Eigen::MatrixXd& Z, const Eigen::VectorXd& y) {
    int rows = Z.rows();
    int cols = Z.cols();
    Eigen::MatrixXd d2Z_dy2 = Eigen::MatrixXd::Zero(rows, cols);

    for (int i = 1; i < rows - 1; ++i) {
        double dy_prev = y(i) - y(i - 1);
        double dy_next = y(i + 1) - y(i);
        for (int j = 0; j < cols; ++j) {
            double z_prev = Z(i - 1, j);
            double z_curr = Z(i, j);
            double z_next = Z(i + 1, j);
            double dy = 0.5 * (dy_prev + dy_next);
            d2Z_dy2(i, j) = (z_next - 2 * z_curr + z_prev) / (dy * dy);
        }
    }

    return d2Z_dy2;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> gradient(const Eigen::MatrixXd& Z, const Eigen::VectorXd& x, const Eigen::VectorXd& y){
    int Nx = x.size();
    int Ny = y.size();

    Eigen::MatrixXd dz_dx = Eigen::MatrixXd::Zero(Ny, Nx);
    Eigen::MatrixXd dz_dy = Eigen::MatrixXd::Zero(Ny, Nx);

    for (int i = 1; i < Ny - 1; ++i) {
        for (int j = 1; j < Nx - 1; ++j) {
            dz_dx(i, j) = (Z(i, j + 1) - Z(i, j - 1)) / (x[j + 1] - x[j - 1]);
            dz_dy(i, j) = (Z(i + 1, j) - Z(i - 1, j)) / (y[i + 1] - y[i - 1]);
        }
    }


    return {dz_dx, dz_dy};
}


/**
    * Approximate maximal mean curvature
    * @param Z: Eigen-matrix with z values in grid
    * @param x: Eigen-vector of x values
    * @param y: Eigen-vector of y values
    * 
    * @return Approximate mean curvature
    */
double approx_peak_curvature(const Eigen::MatrixXd& dz_dx_mat, const Eigen::MatrixXd& dz_dy_mat, const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    // Find peak index
    // Eigen::Index maxRow, maxCol;
    // Z.maxCoeff(&maxRow, &maxCol);

    // // Compute second derivatives
    // Eigen::MatrixXd dz_dx2 = second_derivative_x(Z, x);
    // Eigen::MatrixXd dz_dy2 = second_derivative_y(Z, y);

    // // Evaluate curvature at peak
    // double curv_xx = dz_dx2(maxRow, maxCol);
    // double curv_yy = dz_dy2(maxRow, maxCol);

    // double mean_curv = 0.5 * (curv_xx + curv_yy);
    
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 1>> dz_dx(dz_dx_mat.data(), dz_dx_mat.size());
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 1>> dz_dy(dz_dy_mat.data(), dz_dy_mat.size());

    // Eigen::ArrayXd dz_dx = Eigen::Map<Eigen::ArrayXd>(dz_dx_mat.data(), dz_dx_mat.size());  
    // Eigen::ArrayXd dz_dy = Eigen::Map<Eigen::ArrayXd>(dz_dy_mat.data(), dz_dy_mat.size());

    // Compute second derivatives
    auto [dz_dx_dx_mat, dz_dx_dy_mat] = gradient(dz_dx_mat, x, y);
    auto [dz_dy_dx_mat, dz_dy_dy_mat] = gradient(dz_dy_mat, x, y);

    Eigen::ArrayXd dz_dx_dx = Eigen::Map<Eigen::ArrayXd>(dz_dx_dx_mat.data(), dz_dx_dx_mat.size());
    Eigen::ArrayXd dz_dx_dy = Eigen::Map<Eigen::ArrayXd>(dz_dx_dy_mat.data(), dz_dx_dy_mat.size());
    Eigen::ArrayXd dz_dy_dx = Eigen::Map<Eigen::ArrayXd>(dz_dy_dx_mat.data(), dz_dy_dx_mat.size());
    Eigen::ArrayXd dz_dy_dy = Eigen::Map<Eigen::ArrayXd>(dz_dy_dy_mat.data(), dz_dy_dy_mat.size());

    Eigen::ArrayXd nominator = ((1 + dz_dx * dz_dx) * dz_dy_dy
                                - 2 * dz_dx * dz_dy * dz_dx_dy 
                                + (1 + dz_dy * dz_dy) * dz_dx_dx);
    Eigen::ArrayXd denominator = (sqrt(1 + dz_dx* dz_dx + dz_dy * dz_dy) 
                                    * (1 + dz_dx * dz_dx + dz_dy * dz_dy));
    Eigen::ArrayXd mean_curv_arr = 0.5 * nominator / denominator;

    double mean_curv = mean_curv_arr.matrix().maxCoeff();

    return mean_curv;
}


void MASSystem::generateBumpSurface(const char* jsonPath) {
    // ------------- Load json file --------------
    // Open the file
    std::ifstream ifs(jsonPath);
    if (!ifs.is_open()) {
        std::cerr << "Could not open JSON file.\n";
        return;
    }
    
    // Wrap the input stream
    rapidjson::IStreamWrapper isw(ifs);
    
    // Parse the JSON
    rapidjson::Document doc;
    doc.ParseStream(isw);
    
    if (doc.HasParseError()) {
        std::cerr << "Error parsing JSON.\n";
        return;
    }
    
    // Load values
    int resolution = doc["resolution"].GetInt();
    double xdim = doc["halfWidth_x"].GetDouble();
    double ydim = doc["halfWidth_y"].GetDouble();
    // double zdim = doc["halfWidth_z"].GetDouble();
    const auto& bumpData = doc["bumpData"];

    // Read incidence vector
    const auto& kjson = doc["k"];
    Eigen::Vector3d k;
    for (rapidjson::SizeType i = 0; i < kjson.Size(); ++i) {
        k(i) = kjson[i].GetDouble();
    }

    this->kinc_ = k;
    
    // Read polarizations
    const auto& betas = doc["betas"];
    int B = static_cast<int>(betas.Size());
    Eigen::VectorXd beta_vec(B);
    for (int i = 0; i < B; ++i) {
        beta_vec(i) = betas[i].GetDouble();   // constants.pi / 2 - 
    }
    this->polarizations_ = beta_vec;

    // ------------- Decide number of testpoints --------------
    constants.setWavelength(2 * constants.pi / doc["omega"].GetDouble());
    double lambda0 = constants.getWavelength();
    int Nx = std::ceil(2 * testpts_pr_lambda_ * 2 * xdim / lambda0);
    int Ny = std::ceil(2 * testpts_pr_lambda_ * 2 * ydim / lambda0);

    std::cout << "Nx: " << Nx << std::endl;
    std::cout << "Ny: " << Ny << std::endl;
    

    // ------------- Generate grid -------------
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(Nx, -xdim, xdim);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(Ny, -ydim, ydim);

    Eigen::MatrixXd X(Ny, Nx), Y(Ny, Nx);
    for (int i = 0; i < Ny; ++i) {  // Works as meshgrid
        X.row(i) = x.transpose();
    }
    
    for (int i = 0; i < Nx; ++i) {  // Works as meshgrid
        Y.col(i) = y;
    }
    
    Eigen::MatrixXd Z(Ny,Nx);
    // Z = Eigen::MatrixXd::Zero(N,N);
    Z = Eigen::MatrixXd::Constant(Ny, Nx, -1.5);

    // Iterate over bumpData array
    for (rapidjson::SizeType i = 0; i < bumpData.Size(); ++i) {
        const auto& bump = bumpData[i];
        double x0 = bump["x0"].GetDouble();
        double y0 = bump["y0"].GetDouble();
        double height = bump["height"].GetDouble();
        double sigma = bump["sigma"].GetDouble();
        
        // Add bumps to data
        Z += (height * exp(-((X.array() - x0)*(X.array() - x0) 
                            + (Y.array() - y0)*(Y.array() - y0)) / (2 * sigma * sigma))).matrix();
        
    }

    // ------------- Flatten grid and remove edge points -------------
    Eigen::VectorXd X_flat = Eigen::Map<const Eigen::VectorXd>(X.data(), X.size());
    Eigen::VectorXd Y_flat = Eigen::Map<const Eigen::VectorXd>(Y.data(), Y.size());
    Eigen::VectorXd Z_flat = Eigen::Map<const Eigen::VectorXd>(Z.data(), Z.size());

    std::vector<int> mask_indices;
    for (int i = 0; i < X_flat.size(); ++i) {
        if (X_flat(i) > x(0) && X_flat(i) < x(Nx - 1) &&
            Y_flat(i) > y(0) && Y_flat(i) < y(Ny - 1)) {
            mask_indices.push_back(i);
        }
    }

    int total_pts = static_cast<int>(mask_indices.size());
    Eigen::MatrixXd interior_points(total_pts, 3);
    for (int i = 0; i < total_pts; ++i) {
        int idx = mask_indices[i];
        interior_points.row(i) << X_flat(idx), Y_flat(idx), Z_flat(idx);
    }

    // ------------- Sample test points -------------
    std::vector<int> idx_test;
    for (int i = 0; i < total_pts; i += 2)
        idx_test.push_back(i);

    int N_test_actual = static_cast<int>(idx_test.size());
    Eigen::MatrixX3d test_points(N_test_actual, 3);
    for (int i = 0; i < N_test_actual; ++i)
        test_points.row(i) = interior_points.row(idx_test[i]);

    this->points_ = test_points;

    // ------------- Compute gradients (central differences) -------------
    // Eigen::MatrixXd dz_dx = Eigen::MatrixXd::Zero(Ny, Nx);
    // Eigen::MatrixXd dz_dy = Eigen::MatrixXd::Zero(Ny, Nx);

    // for (int i = 1; i < Ny - 1; ++i) {
    //     for (int j = 1; j < Nx - 1; ++j) {
    //         dz_dx(i, j) = (Z(i, j + 1) - Z(i, j - 1)) / (x[j + 1] - x[j - 1]);
    //         dz_dy(i, j) = (Z(i + 1, j) - Z(i - 1, j)) / (y[i + 1] - y[i - 1]);
    //     }
    // }


    auto [dz_dx, dz_dy] = gradient(Z, x, y);

    Eigen::VectorXd dz_dx_flat = Eigen::Map<Eigen::VectorXd>(dz_dx.data(), dz_dx.size());
    Eigen::VectorXd dz_dy_flat = Eigen::Map<Eigen::VectorXd>(dz_dy.data(), dz_dy.size());


    Eigen::MatrixXd normals(total_pts, 3);
    for (int i = 0; i < total_pts; ++i) {
        int idx = mask_indices[i];
        Eigen::Vector3d n(-dz_dx_flat(idx), -dz_dy_flat(idx), 1.0);
        normals.row(i) = n.normalized();
    }

    Eigen::MatrixXd test_normals(N_test_actual, 3);
    for (int i = 0; i < N_test_actual; ++i)
        test_normals.row(i) = normals.row(idx_test[i]);

    this->normals_ = test_normals;

    // ------------- Tangent vectors -------------
    Eigen::MatrixX3d tangent1(N_test_actual, 3);
    Eigen::MatrixX3d tangent2(N_test_actual, 3);

    for (int i = 0; i < N_test_actual; ++i) {
        Eigen::Vector3d n = test_normals.row(i);
        Eigen::Vector3d ref = (std::abs(n(2)) < 0.9) ? Eigen::Vector3d(0, 0, 1) : Eigen::Vector3d(1, 0, 0);
        Eigen::Vector3d t1 = ref - ref.dot(n) * n;
        t1.normalize();
        Eigen::Vector3d t2 = n.cross(t1);

        tangent1.row(i) = t1;
        tangent2.row(i) = t2;
    }

    this->tau1_ = tangent1;
    this->tau2_ = tangent2;

    // ------------- Auxiliary points -------------
    // Approximate mean curvature and calculate distance
    auto mean_curv = approx_peak_curvature(dz_dx, dz_dy, x, y);
    double radius = 1 / abs(mean_curv);

    double d = (1 - constants.alpha) * radius;  // Distance from surface

    // Calculate points
    std::vector<int> aux_test_indices;
    for (int i = 0; i < N_test_actual; i += 2)
        aux_test_indices.push_back(i);

    this->aux_test_indices_ = aux_test_indices;

    int N_aux = static_cast<int>(aux_test_indices.size());
    Eigen::MatrixX3d aux_points_int(N_aux, 3);
    Eigen::MatrixX3d aux_points_ext(N_aux, 3);

    for (int i = 0; i < N_aux; ++i) {
        int idx = aux_test_indices[i];
        Eigen::Vector3d base = test_points.row(idx);
        Eigen::Vector3d n = test_normals.row(idx);

        aux_points_int.row(i) = base - d * n;
        aux_points_ext.row(i) = base + d * n;
    }

    this->aux_int_ = aux_points_int;
    this->aux_ext_ = aux_points_ext;
}


void MASSystem::generateGPSurface(const char* jsonPath) {
    std::cout << "Sorry, not implemented yet" << std::endl;
}


// const std::vector<std::shared_ptr<FieldCalculatorUPW>>& MASSystem::getIncidentField(){
//     std::vector<std::shared_ptr<FieldCalculatorUPW>>UPWs;

//     int N = static_cast<int>(polarizations_.size());

//     for (int i = 0; i < N; ++i){
//         UPWs.emplace_back(std::make_shared<FieldCalculatorUPW>(kinc_, 1.0, polarizations_(i), constants_));
//     }

//     return UPWs;
// }