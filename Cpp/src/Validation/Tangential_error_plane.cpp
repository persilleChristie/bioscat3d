#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include "rapidjson/document.h" 
#include "rapidjson/istreamwrapper.h"
#include "../../lib/Utils/Constants.h"
#include "../../lib/Utils/ConstantsModel.h"
#include "../../lib/Forward/MASSystem.h"
#include "../../lib/Forward/FieldCalculatorTotal.h"
#include "../../lib/Utils/UtilsExport.h"
#include "../../lib/Utils/UtilsPybind.h"


Eigen::VectorXd calculateTangentialError(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const Eigen::MatrixXd& Z,
                                            const Eigen::VectorXd& wavelengths, const Eigen::VectorXd& polarizations){


}