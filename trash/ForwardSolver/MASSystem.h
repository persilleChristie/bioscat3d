#ifndef MAS_SYSTEM_H
#define MAS_SYSTEM_H

#include "Constants.h"
#include "FieldCalculatorUPW.h"
#include <Eigen/Dense>

/**
    * Class MASSystem, storing a testsurface and two auxillariy surfaces for running MAS. It also stores all 
    * parameters for the incident field.
    *  
    * @param jsonPath char with path to json file including the parameters needed for each surface:
    *       - Bump needs 'resolution', 'halfWidth_x', 'halfWidth_y', 'halfWidth_z', 'bumpData' (hereunder 
    *                   'x0', 'y0', 'height' and 'sigma'), 'k' (incidence vector) and 'betas' (list of polarizations)
    *       - GP needs ...
    * @param surfaceType string indicating type of surface - allowed types are 'Bump' and 'GP'
    * @param constants updated Constants file
*/
class MASSystem {
public:
    MASSystem(const std::string surfaceType, const char* jsonPath); 

    // const std::vector<std::shared_ptr<FieldCalculatorUPW>>& getIncidentField();
    const Eigen::Vector3d& getKinc() const          {return kinc_;};
    const Eigen::VectorXd& getPolarizations() const {return polarizations_;};

    void setPoints(Eigen::MatrixX3d points);
    const Eigen::MatrixX3d& getPoints() const   {return points_;};
    const Eigen::MatrixX3d& getNormals() const  {return normals_;};
    const Eigen::MatrixX3d& getTau1() const     {return tau1_;};
    const Eigen::MatrixX3d& getTau2() const     {return tau2_;};

    const Eigen::MatrixX3d& getInterior() const {return aux_int_;};
    const Eigen::MatrixX3d& getExterior() const {return aux_ext_;};
    const std::vector<int>& getIndecees() const {return aux_test_indices_;};

private:
    // Constants constants_;
    Eigen::Vector3d kinc_;
    Eigen::VectorXd polarizations_;

    Eigen::MatrixX3d points_;
    Eigen::MatrixX3d normals_;
    Eigen::MatrixX3d tau1_;
    Eigen::MatrixX3d tau2_;

    Eigen::MatrixX3d aux_int_;
    Eigen::MatrixX3d aux_ext_;
    std::vector<int> aux_test_indices_;

    void generateBumpSurface(const char* jsonPath);
    void generateGPSurface(const char* jsonPath);
};

#endif // MAS_SURFACE_H
