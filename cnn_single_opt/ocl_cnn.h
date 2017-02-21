#pragma once

#include <string>
#include <map>
#include <vector>

#include "xcl.h"

#include "DataBlob.h"

class oclCNN
{
public:
    oclCNN(std::map<std::string, DataBlob<float>> & model, xcl_world &  world, 
           const std::vector<cl_kernel> & kernels); 
    ~oclCNN();

    DataBlob<float> runImg(DataBlob<float> & img);
private:
    std::map<std::string, DataBlob<float>> & m_model;
    xcl_world & m_world;

    cl_kernel conv1; // Convolution kernel 1
    cl_kernel conv2; // Convolution kernel 2
    cl_kernel mpool1;// Max Pooling kernel 1
    cl_kernel mpool2;// Max Pooling kernel 2
    cl_kernel fc1;   // Fully Connected kernel 1
    cl_kernel fc2;   // Fully Connected kernel 2
    cl_kernel soft; // Softmax kernel

    // Required Work-group sizes
    std::vector<size_t> conv1_wg;
    std::vector<size_t> conv2_wg;
    std::vector<size_t> mpool1_wg;
    std::vector<size_t> mpool2_wg;
    std::vector<size_t> fc1_wg;
    std::vector<size_t> fc2_wg;
    std::vector<size_t> soft_wg;

    // CNN Model Parameters
    clDataBlob<float> cl_wc1;
    clDataBlob<float> cl_bc1;
    clDataBlob<float> cl_wc2;
    clDataBlob<float> cl_bc2;
    clDataBlob<float> cl_wd1;
    clDataBlob<float> cl_bd1;
    clDataBlob<float> cl_wdo;
    clDataBlob<float> cl_bdo;

    // CNN Intermediate Data
    clDataBlob<float> conv1_out;
    clDataBlob<float> pool1_out;
    clDataBlob<float> conv2_out;
    clDataBlob<float> pool2_out;
    clDataBlob<float> dens1_out;
    clDataBlob<float> class_out;
    clDataBlob<float> softm_out;
};
