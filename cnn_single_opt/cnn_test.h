#pragma once

#include <vector>
#include <string>

// OpenCL includes
#include "xcl.h"

class cnn_test
{
public:
    cnn_test(xcl_world & world, const char * clFilename,  bool isBinary);
    ~cnn_test();

    float test_img();
private:
    std::vector<cl_kernel> kernels;
    xcl_world & m_world;
};
