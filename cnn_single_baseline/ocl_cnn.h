#pragma once

#include <string>
#include <map>
#include <vector>

#include "xcl.h"

#include "DataBlob.h"


DataBlob<float> ocl_cnn_img_test(DataBlob<float> & img, std::map<std::string, DataBlob<float>> & model, 
                                 xcl_world & world, const std::vector<cl_kernel> & kernels);
