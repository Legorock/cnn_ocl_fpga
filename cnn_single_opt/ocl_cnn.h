#pragma once

#include <string>
#include <map>
#include <vector>

#include "xcl.h"

#include "DataBlob.h"


DataBlob<float> ocl_cnn_img_test(DataBlob<float> & img, std::map<std::string, DataBlob<float>> & model, 
                                 xcl_world & world, const std::vector<cl_kernel> & kernels);


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

    cl_kernel conv; // Convolution kernel
    cl_kernel mpool;// Max Pooling kernel
    cl_kernel fc;   // Fully Connected kernel
    cl_kernel soft; // Softmax kernel

    // Required Work-group sizes
    std::vector<size_t> conv_wg;
    std::vector<size_t> mpool_wg;
    std::vector<size_t> fc_wg;
    std::vector<size_t> soft_wg;

    clDataBlob<float> cl_wc1;
    clDataBlob<float> cl_bc1;
    clDataBlob<float> cl_wc2;
    clDataBlob<float> cl_bc2;
    clDataBlob<float> cl_wd1;
    clDataBlob<float> cl_bd1;
    clDataBlob<float> cl_wdo;
    clDataBlob<float> cl_bdo;

    clDataBlob<float> conv1_out;
    clDataBlob<float> pool1_out;
    clDataBlob<float> conv2_out;
    clDataBlob<float> pool2_out;
    clDataBlob<float> dens1_out;
    clDataBlob<float> class_out;
    clDataBlob<float> softm_out;
};
