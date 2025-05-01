#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include "rapidjson/document.h" 
#include "rapidjson/istreamwrapper.h"
#include "../../ForwardSolver/FieldCalculatorTotal.h"
#include "../../ForwardSolver/MASSystem.h"
#include "../../ForwardSolver/SurfacePlane.h"
#include "../../ForwardSolver/UtilsExport.h"
#include "../../ForwardSolver/Constants.h"

int main(){
    const char* jsonPath = "surfaceParams.json";

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
    double cell_x = doc["halfWidth_x"].GetDouble();
    double cell_y = doc["halfWidth_y"].GetDouble();
    double cell_z = doc["halfWidth_z"].GetDouble();
    const auto& bumpData = doc["bumpData"];

    MASSystem masSystem(jsonPath, "Bump", constants);

    FieldCalculatorTotal field(masSystem, constants);

    // Top
    Eigen::Vector3d Cornerpoint (-cell_x, -cell_y, 0.5*cell_z - pml_thickness - 0.1);
    Eigen::Vector3d basis1 (1,0,0), basis2 (0,1,0);
    SurfacePlane top(Cornerpoint, basis1, basis2, size1, size2, resolution);


    // Bottom
    Cornerpoint << -cell_x, -cell_y, -0.5*cell_z + pml_thickness + 0.1;
    SurfacePlane bottom(Cornerpoint, basis1, basis2, size1, size2, resolution);


    // Left
    Cornerpoint << -0.5*cell_x + pml_thickness + 0.1, -cell_y, -cell_z;
    basis1 << 0,1,0;
    basis2 << 0,0,1;
    SurfacePlane left(Cornerpoint, basis1, basis2, size1, size2, resolution);

    
    // Right
    Cornerpoint << 0.5*cell_x - pml_thickness - 0.1, -cell_y, -cell_z;
    SurfacePlane right(Cornerpoint, basis1, basis2, size1, size2, resolution);

    

    // Back
    Cornerpoint << -cell_x, -0.5*cell_y + pml_thickness + 0.1, -cell_z;
    basis1 << 1,0,0;
    basis2 << 0,0,1;
    SurfacePlane back(Cornerpoint, basis1, basis2, size1, size2, resolution);

    // Front
    Cornerpoint << -cell_x, 0.5*cell_y - pml_thickness - 0.1, -cell_z;
    SurfacePlane front(Cornerpoint, basis1, basis2, size1, size2, resolution);


    Export::saveRealVectorCSV("FilesCSV/powers_polarization.csv", powers);



    return 0;
}