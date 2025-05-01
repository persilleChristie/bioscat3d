#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <map>
#include "rapidjson/document.h" 
#include "rapidjson/istreamwrapper.h"
#include "../../ForwardSolver/FieldCalculatorTotal.h"
#include "../../ForwardSolver/MASSystem.h"
#include "../../ForwardSolver/SurfacePlane.h"
#include "../../ForwardSolver/UtilsExport.h"
#include "../../ForwardSolver/Constants.h"



Constants constants;

int main(){

    const char* jsonPath = "surfaceParams.json";

    // Open the file
    std::ifstream ifs(jsonPath);
    if (!ifs.is_open()) {
        std::cerr << "Could not open JSON file.\n";
        return 1;
    }
    
    // Wrap the input stream
    rapidjson::IStreamWrapper isw(ifs);
    
    // Parse the JSON
    rapidjson::Document doc;
    doc.ParseStream(isw);
    
    if (doc.HasParseError()) {
        std::cerr << "Error parsing JSON.\n";
        return 1;
    }
    
    // Load values
    int resolution = doc["resolution"].GetInt();
    double halfwidth_x = doc["halfWidth_x"].GetDouble();
    double halfwidth_y = doc["halfWidth_y"].GetDouble();
    double halfwidth_z = doc["halfWidth_z"].GetDouble();
    double pml_thickness = doc["pml_thickness"].GetDouble();
    double omega = doc["omega"].GetDouble();

    double cell_x = 2*halfwidth_x + 2*pml_thickness;
    double cell_y = 2*halfwidth_y + 2*pml_thickness;
    double cell_z = 2*halfwidth_z + 2*pml_thickness;

    double size1 = cell_x, size2 = cell_y;

    constants.setWavelength(2 * constants.pi / omega);

    // ------------ Define planes -------------
    std::map<std::string, SurfacePlane> test_planes;

    // Top
    Eigen::Vector3d Cornerpoint (-cell_x, -cell_y, 0.5*cell_z - pml_thickness - 0.1);
    Eigen::Vector3d basis1 (1,0,0), basis2 (0,1,0);
    SurfacePlane top(Cornerpoint, basis1, basis2, size1, size2, resolution);
    test_planes.insert(std::pair<std::string, SurfacePlane>("top", top));
    // std::cout << "top" << top.getNormals().row(0) << std::endl;

    // Bottom
    Cornerpoint << -cell_x, -cell_y, -0.5*cell_z + pml_thickness + 0.1;
    SurfacePlane bottom(Cornerpoint, basis1, basis2, size1, size2, resolution);
    test_planes.insert(std::pair<std::string, SurfacePlane>("bottom", bottom));
    // std::cout << "bottom" << bottom.getNormals().row(0) << std::endl;


    // Left
    Cornerpoint << -0.5*cell_x + pml_thickness + 0.1, -cell_y, -cell_z;
    basis1 << 0,1,0;
    basis2 << 0,0,1;
    SurfacePlane left(Cornerpoint, basis1, basis2, size1, size2, resolution);
    test_planes.insert(std::pair<std::string, SurfacePlane>("left", left));
    // std::cout << "left" << left.getNormals().row(0) << std::endl;

    
    // Right
    Cornerpoint << 0.5*cell_x - pml_thickness - 0.1, -cell_y, -cell_z;
    SurfacePlane right(Cornerpoint, basis1, basis2, size1, size2, resolution);
    test_planes.insert(std::pair<std::string, SurfacePlane>("right", right));
    // std::cout << "right" << right.getNormals().row(0) << std::endl;


    // Back
    Cornerpoint << -cell_x, -0.5*cell_y + pml_thickness + 0.1, -cell_z;
    basis1 << 1,0,0;
    basis2 << 0,0,1;
    SurfacePlane back(Cornerpoint, basis2, basis1, size1, size2, resolution);
    test_planes.insert(std::pair<std::string, SurfacePlane>("back", back));
    // std::cout << "back" << back.getNormals().row(0) << std::endl;


    // Front
    Cornerpoint << -cell_x, 0.5*cell_y - pml_thickness - 0.1, -cell_z;
    SurfacePlane front(Cornerpoint, basis2, basis1, size1, size2, resolution);
    test_planes.insert(std::pair<std::string, SurfacePlane>("front", front));
    // std::cout << "front" << front.getNormals().row(0) << std::endl;




    // -------------- Calculate powers ------------------
    MASSystem masSystem(jsonPath, "Bump", constants);

    FieldCalculatorTotal field(masSystem, constants);

    std::string filename1, filename2;

    for (auto it : test_planes){
        auto [powers, integrand] = field.computePower(it.second);

        filename1 = "FilesCSV/powers_polarization_" + it.first + ".csv";
        filename2 = "FilesCSV/integrand_beta0_" + it.first + ".csv";

        Export::saveRealVectorCSV(filename1, powers);
        Export::saveRealMatrixCSV(filename2, integrand);
    }
    



    return 0;
}