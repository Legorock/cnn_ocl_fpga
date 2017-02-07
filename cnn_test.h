#pragma once

#include <vector>
#include <string>

// OpenCL includes
#include "xcl.h"

#include "DataBlob.h"

typedef DataBlob<float> Data;

class cnn_test
{
public:
    cnn_test(xcl_world & world, const char * clFilename,  bool isBinary);
    ~cnn_test();

    float test_img();
private:
    std::vector<cl_kernel> kernels;
    xcl_world & test_world;
private:
    Data seq_img_test(const Data& img);
    Data ocl_img_test(Data& img);
    
};
