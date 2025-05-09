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


/**
    * Approximate gradient in x and y using central differences for interior points, and forward/backward 
    * differences for interior points 
    * (to mimick numpy's gradient, see https://numpy.org/doc/2.1/reference/generated/numpy.gradient.html)
    * 
    * @param Z: Matrix with z values in grid (x horizonthal, y vertical)
    * @param x: Vector of x values
    * @param y: Vector of y values
    * 
    * @return Matrix of the second derivative
    */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> gradient(const Eigen::MatrixXd& Z, 
                                                        const Eigen::VectorXd& x, const Eigen::VectorXd& y){
    int Nx = x.size();
    int Ny = y.size();

    Eigen::MatrixXd dz_dx = Eigen::MatrixXd::Zero(Ny, Nx);
    Eigen::MatrixXd dz_dy = Eigen::MatrixXd::Zero(Ny, Nx);

    for (int i = 1; i < Ny - 1; ++i) {
        // Central difference for interior points
        for (int j = 1; j < Nx - 1; ++j) {
            dz_dx(i, j) = (Z(i, j + 1) - Z(i, j - 1)) / (x[j + 1] - x[j - 1]);
            dz_dy(i, j) = (Z(i + 1, j) - Z(i - 1, j)) / (y[i + 1] - y[i - 1]);
        }
    }

    // Forward/backward difference for exterior points x-direction
    for (int i = 0; i < Ny; ++i){
        dz_dx(i, 0) = (Z(i, 1) - Z(i, 0)) / (x[1] - x[0]);
        dz_dx(i, Nx - 1) = (Z(i, Nx - 1) - Z(i, Nx - 2)) / (x[Nx - 1] - x[Nx - 2]);
    }

    // Forward/backward difference for exterior points y-direction
    for (int j = 0; j < Nx; ++j){
        dz_dy(0, j) = (Z(1, j) - Z(0, j))/ (y[1] - y[0]);
        dz_dy(Ny - 1, j) = (Z(Ny - 1, j) - Z(Ny - 2, j))/ (y[Ny - 1] - y[Ny - 2]);
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
double approx_max_curvature(const Eigen::MatrixXd& dz_dx_mat, const Eigen::MatrixXd& dz_dy_mat, const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 1>> dz_dx(dz_dx_mat.data(), dz_dx_mat.size());
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 1>> dz_dy(dz_dy_mat.data(), dz_dy_mat.size());

    auto [dz_dx_dx, dz_dx_dy] = gradient(dz_dx_mat, x, y);
    auto [dz_dy_dx, dz_dy_dy] = gradient(dz_dy_mat, x, y);

    Eigen::ArrayXXd fx = dz_dx_mat.array();
    Eigen::ArrayXXd fy = dz_dy_mat.array();
    Eigen::ArrayXXd fxx = dz_dx_dx.array();
    Eigen::ArrayXXd fyy = dz_dy_dy.array();
    Eigen::ArrayXXd fxy = dz_dx_dy.array(); // assumed symmetric with dy_dx

    Eigen::ArrayXXd fx2 = fx.square();
    Eigen::ArrayXXd fy2 = fy.square();
    Eigen::ArrayXXd grad_sq = 1.0 + fx2 + fy2;
    Eigen::ArrayXXd sqrt_grad_sq = grad_sq.sqrt();
    Eigen::ArrayXXd denom = sqrt_grad_sq * grad_sq;    

    Eigen::ArrayXXd numerator = (1.0 + fx2) * fyy - 2.0 * fx * fy * fxy + (1.0 + fy2) * fxx;
    Eigen::ArrayXXd mean_curv = 0.5 * numerator / denom;

    std::cout << "fxx = " << fxx << std::endl;
    std::cout << "fyy = " << fyy << std::endl;
    std::cout << "fx2 = " << fx2 << std::endl;
    std::cout << "fy2 = " << fy2 << std::endl;

    
    std::cout << "denom = " << denom << std::endl;
    std::cout << "numerator = " << numerator << std::endl;

    return mean_curv.maxCoeff();
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
    int Nx = std::ceil(2 * auxpts_pr_lambda_ * 2 * xdim / lambda0);
    int Ny = std::ceil(2 * auxpts_pr_lambda_ * 2 * ydim / lambda0);
    

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
    int total_pts = Z.size();

    Eigen::VectorXd X_flat = Eigen::Map<const Eigen::VectorXd>(X.data(), total_pts);
    Eigen::VectorXd Y_flat = Eigen::Map<const Eigen::VectorXd>(Y.data(), total_pts);
    Eigen::VectorXd Z_flat = Eigen::Map<const Eigen::VectorXd>(Z.data(), total_pts);

    // std::vector<int> mask_indices;
    // for (int i = 0; i < X_flat.size(); ++i) {
    //     if (X_flat(i) > x(0) && X_flat(i) < x(Nx - 1) &&
    //         Y_flat(i) > y(0) && Y_flat(i) < y(Ny - 1)) {
    //         mask_indices.push_back(i);
    //     }
    // }

    

    // int total_pts = static_cast<int>(mask_indices.size());
    // Eigen::MatrixXd interior_points(total_pts, 3);
    // for (int i = 0; i < total_pts; ++i) {
    //     int idx = mask_indices[i];
    //     interior_points.row(i) << X_flat(idx), Y_flat(idx), Z_flat(idx);
    // }

    // ------------- Sample test points -------------
    // std::vector<int> idx_test;
    // for (int i = 0; i < total_pts; i += 2)
    //     {idx_test.push_back(i);

    int N_test_actual = total_pts; // <int>(idx_test.size());

    Eigen::MatrixX3d test_points(N_test_actual, 3);
    test_points.col(0) = X_flat;
    test_points.col(1) = Y_flat;
    test_points.col(2) = Z_flat;
    // for (int i = 0; i < N_test_actual; ++i)
    //     // test_points.row(i) = interior_points.row(i);
    //     {test_points.row(i) << X_flat(i), Y_flat(i), Z_flat(i);}


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

    Eigen::VectorXd dz_dx_flat = Eigen::Map<Eigen::VectorXd>(dz_dx.data(), total_pts);
    Eigen::VectorXd dz_dy_flat = Eigen::Map<Eigen::VectorXd>(dz_dy.data(), total_pts);
    Eigen::Vector3d n;
    Eigen::MatrixXd normals(total_pts, 3);

    for (int i = 0; i < total_pts; ++i) {
        // int idx = mask_indices[i];

        n << -dz_dx_flat(i), -dz_dy_flat(i), 1.0;
        normals.row(i) = n.normalized();
    }

    // Eigen::MatrixXd test_normals(N_test_actual, 3);
    // for (int i = 0; i < N_test_actual; ++i)
    //     test_normals.row(i) = normals.row(idx_test[i]);

    this->normals_ = normals;

    // ------------- Tangent vectors -------------
    Eigen::MatrixX3d tangent1(N_test_actual, 3);
    Eigen::MatrixX3d tangent2(N_test_actual, 3);
    Eigen::Vector3d t1;
    Eigen::Vector3d t2;
    Eigen::Vector3d ref;

    for (int i = 0; i < N_test_actual; ++i) {
        n << normals.row(i).transpose();
        ref << ((std::abs(n(2)) < 0.9) ? Eigen::Vector3d(0, 0, 1) : Eigen::Vector3d(1, 0, 0));
        t1 << ref - ref.dot(n) * n;
        t1.normalize();
        t2 << n.cross(t1);

        tangent1.row(i) = t1;
        tangent2.row(i) = t2;
    }

    this->tau1_ = tangent1;
    this->tau2_ = tangent2;

    // ------------- Auxiliary points -------------
    // Approximate mean curvature and calculate distance
    auto mean_curv = approx_max_curvature(dz_dx, dz_dy, x, y);
    std::cout << "Max mean curvature: " << mean_curv << std::endl;
    double radius = 1 / abs(mean_curv);
    std::cout << "Max mean curvature: " << mean_curv << std::endl;
    
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
        Eigen::Vector3d n = normals.row(idx);

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