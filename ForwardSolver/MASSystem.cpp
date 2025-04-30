#include "MASSystem.h"
#include "Constants.h"
#include "rapidjson/document.h" 
#include "rapidjson/filereadstream.h" 

MASSystem::MASSystem(const char* jsonPath, const std::string surfaceType, Constants constants)
    : constants_(constants)
    {
        if (surfaceType == "Bump"){
            generateBumpSurface(jsonPath);
        } else if (surfaceType == "GP"){
            generateGPSurface(jsonPath);
        } else {
            std::cerr << "[Error] Surface type not recognised! Allowed types are 'Bump' and 'GP'\n";
        }
    }

void MASSystem::generateBumpSurface(const char* jsonPath) {
    // ------------- Load json file --------------
    rapidjson::Document document;
    document.Parse(jsonPath);

    

    // ------------- Generate grid -------------
    int N = resolution_;
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N, -xdim_, xdim_);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(N, -ydim_, ydim_);

    Eigen::MatrixXd X(N, N), Y(N, N);
    for (int i = 0; i < N; ++i) {  // Works as meshgrid
        X.row(i) = x.transpose();
        Y.col(i) = y;
    }

    Eigen::MatrixXd Z;
    Z = Eigen::MatrixXd::Zero(N,N);

    // ------------- Add bumps -------------
    int nr_bumps = x0_.size();

    for (int i = 0; i < nr_bumps; ++i){
        Z += (h_(i) * exp(-((X.array() - x0_(i))*(X.array() - x0_(i)) 
                            + (Y.array() - y0_(i))*(Y.array() - y0_(i))) / (2 * sigma_(i) * sigma_(i)))).matrix();
    }

    // ------------- Decide number of testpoints --------------
    double lambda0 = constants_.getWavelength();
    int N_test = std::ceil(2.0 * testpts_pr_lambda_ * 2 * xdim_ * testpts_pr_lambda_ * 2 * ydim_ / (lambda0 * lambda0));

    // ------------- Flatten grid and remove edge points -------------
    Eigen::VectorXd X_flat = Eigen::Map<const Eigen::VectorXd>(X.data(), X.size());
    Eigen::VectorXd Y_flat = Eigen::Map<const Eigen::VectorXd>(Y.data(), Y.size());
    Eigen::VectorXd Z_flat = Eigen::Map<const Eigen::VectorXd>(Z.data(), Z.size());

    int nx = resolution_;
    int ny = resolution_;

    std::vector<int> mask_indices;
    for (int i = 0; i < X_flat.size(); ++i) {
        if (X_flat(i) > x(0) && X_flat(i) < x(nx - 1) &&
            Y_flat(i) > y(0) && Y_flat(i) < y(ny - 1)) {
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
    Eigen::MatrixXd test_points(N_test_actual, 3);
    for (int i = 0; i < N_test_actual; ++i)
        test_points.row(i) = interior_points.row(idx_test[i]);

    this->points_ = test_points;

    // ------------- Compute gradients (central differences) -------------
    Eigen::MatrixXd dz_dx = Eigen::MatrixXd::Zero(ny, nx);
    Eigen::MatrixXd dz_dy = Eigen::MatrixXd::Zero(ny, nx);

    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            dz_dx(j, i) = (Z(j, i + 1) - Z(j, i - 1)) / (x[i + 1] - x[i - 1]);
            dz_dy(j, i) = (Z(j + 1, i) - Z(j - 1, i)) / (y[j + 1] - y[j - 1]);
        }
    }

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
    Eigen::MatrixXd tangent1(N_test_actual, 3);
    Eigen::MatrixXd tangent2(N_test_actual, 3);

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
    double d = 1 - constants_.alpha;  // Distance from surface
    std::vector<int> aux_test_indices;
    for (int i = 0; i < N_test_actual; i += 2)
        aux_test_indices.push_back(i);

    int N_aux = static_cast<int>(aux_test_indices.size());
    Eigen::MatrixXd aux_points_int(N_aux, 3);
    Eigen::MatrixXd aux_points_ext(N_aux, 3);

    for (int i = 0; i < N_aux; ++i) {
        int idx = aux_test_indices[i];
        Eigen::Vector3d base = test_points.row(idx);
        Eigen::Vector3d n = test_normals.row(idx);

        aux_points_int.row(i) = base - d * n;
        aux_points_ext.row(i) = base + d * n;
    }

}


void MASSystem::generateGPSurface(const std::string jsonPath) {
    std::cout << "Sorry, not implemenred yet" << std::endl;
}