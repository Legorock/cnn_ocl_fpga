#pragma once

#include <vector>
#include <map>
#include <string>

// OpenCL includes
#include "xcl.h"


struct Data
{
    std::vector<float> buffer;
    std::vector<std::size_t> dims;
};

typedef std::map<std::string, Data> krnl_data_map;

class cnn_test
{
public:
    cnn_test(xcl_world & world, const char * clFilename,  bool isBinary);
    ~cnn_test();
    float test();
    float test_img();
private:
    std::vector<cl_kernel> kernels;
    xcl_world & test_world;
private:
    krnl_data_map ocl_runs(std::map<std::string, std::vector<Data>>& data);
    void ocl_maxpool2_run(cl_kernel kernel, std::vector<Data>& data);
    void ocl_conv_run(cl_kernel kernel, std::vector<Data>& data);
    void ocl_fc_run(cl_kernel kernel, std::vector<Data>& data);
    void ocl_softmax_run(cl_kernel kernel, std::vector<Data>& data);
};
