#pragma once

#include <vector>
#include <string>

#include <map>

//OpenCL includes
#include "xcl.h"


#include "DataBlob.h"


typedef DataBlob<float> Data;

class cnn_runall
{
public:
    cnn_runall(xcl_world & world, const char * clFilename, bool isBinary);
    ~cnn_runall();

    float run_all();
private:
    std::vector<cl_kernel> kernels;
    xcl_world & m_world;

    std::vector<std::vector<float>> train_imgs;
    std::vector<std::vector<float>> test_imgs;
    std::vector<std::vector<float>> train_labels;
    std::vector<std::vector<float>> test_labels;

    std::map<std::string, Data> model_params;
};
