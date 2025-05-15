#include <fstream> 
#include "MASSystem.h"
#include "Constants.h"
#include "rapidjson/document.h" 
#include "rapidjson/istreamwrapper.h"


MASSystem::MASSystem(const std::string surfaceType, const char* jsonPath)
    {
        if (surfaceType == "Bump"){
            generateBumpSurface(jsonPath);
        } else if (surfaceType == "GP"){
            std::cout << "hello" << std::endl;
            generateGPSurface(jsonPath);
        } else {
            std::cerr << "[Error] Surface type not recognised! Allowed types are 'Bump' and 'GP'\n";
        }

    }

void MASSystem::setPoints(Eigen::MatrixX3d points) {
    this->points_ = points;
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
std::pair<Eigen::VectorXd, Eigen::VectorXd> gradient(const Eigen::VectorXd& Z, 
                                                        const Eigen::VectorXd& X, const Eigen::VectorXd& Y, 
                                                        const int Nx, const int Ny){
    int N = Nx*Ny;
    int k;

    Eigen::VectorXd dz_dx = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd dz_dy = Eigen::VectorXd::Zero(N);
    
    for (int i = 1; i < Ny - 1; ++i) {
        // Central difference for interior points
        for (int j = 1; j < Nx - 1; ++j) {
            k = j * Ny + i;

            dz_dx(k) = (Z(k + Ny) - Z(k - Ny)) / (X(k + Ny) - X(k - Ny));
            dz_dy(k) = (Z(k + 1) - Z(k - 1)) / (Y(k + 1) - Y(k - 1));
        }
    }

    // Forward/backward difference for exterior points x-direction
    for (int i = 0; i < Ny; ++i){
        dz_dx(i) = (Z(i + Ny) - Z(i)) / (X(i + Ny) - X(i));
        dz_dx((Nx - 1) * Ny + i) = (Z((Nx - 1) * Ny + i) - Z((Nx - 2) * Ny + i)) 
                                    / (X((Nx - 1) * Ny + i) - X((Nx - 2) * Ny + i));
    }

    // Forward/backward difference for exterior points y-direction
    for (int j = 0; j < Nx; ++j){
        dz_dy(j * Ny) = (Z(Ny * j + 1) - Z(Ny * j)) / (Y(Ny * j + 1) - Y(Ny * j));
        dz_dy((Ny - 1) + Ny * j) = (Z((Ny - 1) + Ny * j) - Z((Ny - 2) + Ny * j))
                                    / (Y((Ny - 1) + Ny * j) - Y((Ny - 2) + Ny * j));
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
double approx_max_curvature(const Eigen::VectorXd& dz_dx, const Eigen::VectorXd& dz_dy, 
                                const Eigen::VectorXd& X, const Eigen::VectorXd& Y,
                                const int Nx, const int Ny) {
    auto [dz_dx_dx, dz_dx_dy] = gradient(dz_dx, X, Y, Nx, Ny);
    auto [dz_dy_dx, dz_dy_dy] = gradient(dz_dy, X, Y, Nx, Ny);

    Eigen::ArrayXd fx = dz_dx.array();
    Eigen::ArrayXd fy = dz_dy.array();
    Eigen::ArrayXd fxx = dz_dx_dx.array();
    Eigen::ArrayXd fyy = dz_dy_dy.array();
    Eigen::ArrayXd fxy = dz_dx_dy.array(); // assumed symmetric with dy_dx

    Eigen::ArrayXd fx2 = fx.square();
    Eigen::ArrayXd fy2 = fy.square();
    Eigen::ArrayXd grad_sq = 1.0 + fx2 + fy2;
    Eigen::ArrayXd sqrt_grad_sq = grad_sq.sqrt();
    Eigen::ArrayXd denom = sqrt_grad_sq * grad_sq;    

    Eigen::ArrayXd numerator = (1.0 + fx2) * fyy - 2.0 * fx * fy * fxy + (1.0 + fy2) * fxx;
    Eigen::ArrayXd mean_curv = abs(0.5 * numerator / denom);

    int maxRow;

    // std::cout << "mean curvature = " << minus_mean_curv << std::endl;
    std::cout << "fx = " << fx << std::endl;
    std::cout << "fy = " << fy << std::endl;
    // std::cout << "fxx = " << fxx << std::endl;
    // std::cout << "fyy = " << fyy << std::endl;
    // std::cout << "fxy = " << fxy << std::endl;
    // std::cout << "fx2 = " << fx2 << std::endl;
    // std::cout << "fy2 = " << fy2 << std::endl;
    // std::cout << "numerator = " << numerator << std::endl;
    // std::cout << "denominator = " << denom << std::endl;

    double mean_curv_max = mean_curv.maxCoeff(&maxRow);

    // std::cout << "X val of max curv: " << X(maxRow) << std::endl;
    // std::cout << "Y val of max curv: " << Y(maxRow) << std::endl; // passer med forventet
    // std::cout << "fxx = " << fxx(maxRow) << std::endl;
    // std::cout << "fyy = " << fyy(maxRow) << std::endl;
    // std::cout << "fx2 = " << fx2(maxRow) << std::endl;
    // std::cout << "fy2 = " << fy2(maxRow) << std::endl;

    
    // std::cout << "denom = " << denom(maxRow) << std::endl;
    // std::cout << "numerator = " << numerator(maxRow) << std::endl;


    return mean_curv_max;
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
    // int resolution = doc["resolution"].GetInt();
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
    int Nx = std::ceil(2 * constants.auxpts_pr_lambda * 2 * xdim / lambda0);
    int Ny = std::ceil(2 * constants.auxpts_pr_lambda * 2 * ydim / lambda0);
    

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
    Z = Eigen::MatrixXd::Constant(Ny, Nx, 0);

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
    
    // int maxRow, maxCol;
    // Z.maxCoeff(&maxRow, &maxCol);
    // std::cout << "X val of peak: " << X(maxRow, maxCol) << std::endl;
    // std::cout << "Y val of peak: " << Y(maxRow, maxCol) << std::endl;

    int N_test_actual = total_pts; // <int>(idx_test.size());

    Eigen::MatrixX3d test_points(N_test_actual, 3);
    test_points.col(0) = X_flat;
    test_points.col(1) = Y_flat;
    test_points.col(2) = Z_flat;
    // for (int i = 0; i < N_test_actual; ++i)
    //     // test_points.row(i) = interior_points.row(i);
    //     {test_points.row(i) << X_flat(i), Y_flat(i), Z_flat(i);}
   

    this->points_ = test_points;

    // ------------- Normal vectors -------------
    auto [dz_dx, dz_dy] = gradient(Z_flat, X_flat, Y_flat, Nx, Ny);

    Eigen::MatrixXd normals(total_pts, 3);
    normals.col(0) = -dz_dx;
    normals.col(1) = -dz_dy;
    normals.col(2) = Eigen::VectorXd::Constant(total_pts, 1.0);

    this->normals_ = normals.rowwise().normalized();

    // ------------- Tangent vectors -------------
    Eigen::MatrixX3d tangent1(N_test_actual, 3);
    Eigen::MatrixX3d tangent2(N_test_actual, 3);
    Eigen::Vector3d t1, t2, n;
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
    auto mean_curv = approx_max_curvature(dz_dx, dz_dy, X_flat, Y_flat, Nx, Ny);
    std::cout << "Max mean curvature: " << mean_curv << std::endl;
    // double mean_curv = -0.6657612170;
    double radius = 1 / abs(mean_curv);
    // double radius = 1.02;

    
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

    // Dimensions
    double xdim = doc["halfWidth_x"].GetDouble();
    double ydim = doc["halfWidth_y"].GetDouble();

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
        beta_vec(i) = betas[i].GetDouble();   
    }
    this->polarizations_ = beta_vec;

    // ------------- Decide number of testpoints --------------
    double lambda0 = 2 * constants.pi / doc["omega"].GetDouble();
    constants.setWavelength(lambda0);
    int Nx = std::ceil(2 * constants.auxpts_pr_lambda * 2 * xdim / lambda0);
    int Ny = std::ceil(2 * constants.auxpts_pr_lambda * 2 * ydim / lambda0);
    int N = Nx * Ny;
    

    // ------------- Generate grid -------------
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(Nx, -xdim, xdim);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(Ny, -ydim, ydim);

    Eigen::MatrixXd Xmat(Ny, Nx), Ymat(Ny, Nx);
   for (int i = 0; i < Ny; ++i) {  // Works as meshgrid
        Xmat.row(i) = x.transpose();
    }
    
    for (int i = 0; i < Nx; ++i) {  // Works as meshgrid
        Ymat.col(i) = y;
    }
    

    Eigen::VectorXd X = Eigen::Map<const Eigen::VectorXd>(Xmat.data(), N);
    Eigen::VectorXd Y = Eigen::Map<const Eigen::VectorXd>(Ymat.data(), N);
    Eigen::VectorXd Z = Eigen::VectorXd::Zero(N);

    Eigen::MatrixX3d points(N,3);
    points.col(0) = X;
    points.col(1) = Y;
    points.col(2) = Z;
    this->points_ = points;
    

    // ------------- Normal vectors -------------
    auto [dz_dx, dz_dy] = gradient(Z, X, Y, Nx, Nx);

    Eigen::MatrixXd normals(N, 3);

    normals.col(0) = -dz_dx;
    normals.col(1) = -dz_dy;
    normals.col(2) = Eigen::VectorXd::Constant(N, 1.0);

    this->normals_ = normals;

    // ------------- Tangent vectors -------------
    Eigen::MatrixX3d tangent1(N, 3);
    Eigen::MatrixX3d tangent2(N, 3);
    Eigen::Vector3d n, t1, t2;
    Eigen::Vector3d ref;

    for (int i = 0; i < N; ++i) {
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
    // auto mean_curv = approx_max_curvature(dz_dx, dz_dy, X, Y, Nx, Nx);
    // std::cout << "Max mean curvature: " << mean_curv << std::endl;
    double mean_curv = -0.6657612170;
    double radius = 1 / abs(mean_curv);
    // double radius = 1.02;

    
    double d = (1 - constants.alpha) * radius;  // Distance from surface

    // Calculate points
    std::vector<int> aux_test_indices;
    for (int i = 0; i < N; i += 2)
        aux_test_indices.push_back(i);

    this->aux_test_indices_ = aux_test_indices;

    int N_aux = static_cast<int>(aux_test_indices.size());
    Eigen::MatrixX3d aux_points_int(N_aux, 3);
    Eigen::MatrixX3d aux_points_ext(N_aux, 3);

    Eigen::Vector3d base;

    for (int i = 0; i < N_aux; ++i) {
        int idx = aux_test_indices[i];
        base = points_.row(idx);
        n = normals.row(idx);

        aux_points_int.row(i) = base - d * n;
        aux_points_ext.row(i) = base + d * n;
    }

    this->aux_int_ = aux_points_int;
    this->aux_ext_ = aux_points_ext;

}
