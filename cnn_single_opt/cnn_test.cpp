#include "cnn_test.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "DataBlob.h"
//#include "genData.h"
#include "seq.h"
#include "seq_cnn.h"
#include "ocl_cnn.h"

#include "ModelImporter.h"
#include "mnist_test_img.h"

#include "helper.h"
#include "Measure.h"

typedef DataBlob<float> Data;

template<typename T>
inline static void print_buf(std::ostream& o, const T *  buf, 
        const std::vector<std::size_t>& dims, const std::size_t curr_dim);

//const std::vector<std::string> kernel_names = {"max_pool2", "conv_local", "softmax_layer", "fc"};
const std::vector<std::string> kernel_names = {"load_model_ocm", "softmax_layer", 
                                               "max_pool1", "max_pool2", 
                                               "conv1", "conv2", "fc1", "fc2"};


static auto absolute = [](const std::vector<float>& seq, const std::vector<float>& ocl)
{
    auto it_seq = seq.begin();
    auto e_seq = seq.end();
    auto it_ocl = ocl.begin();
    auto  e_ocl = ocl.end();
    float diff = 0.0f;
    for(; it_seq != e_seq && it_ocl != e_ocl; ++it_seq, ++it_ocl)
    {
        diff += std::abs( (*it_seq) - (*it_ocl) );
    }
    return diff;
};

inline static Data getTestImg()
{
    Data im = emptyDataBlob<float>({28, 28, 1});
    
    for(std::size_t i = 0; i < 28 * 28; ++i)
    {
        im.buffer[i] = static_cast<float>(mnist_test::img[i]) / 255;
    }
    return im;
}

    template<typename T>
inline static void print_buf(std::ostream& o, const T *  buf, 
        const std::vector<std::size_t>& dims, const std::size_t curr_dim)
{
    if(curr_dim == 0)
    {
        for(std::size_t i = 0; i < dims[curr_dim]; ++i)
        {
            //std::cerr << *(buf + i) << '\t';
            o << *(buf + i) << '\t';
        }
        //std::cerr << '\n';
        o << '\n';
    }
    else
    {
        std::size_t stride = 1;
        for(std::size_t d = curr_dim-1; d != 0; --d)
        {
            stride *= dims[d];
        }
        stride *= dims[0];

        for(std::size_t i = 0; i < dims[curr_dim]; ++i)
        {
            print_buf<T>(o, buf + i * stride, dims, curr_dim-1);
        }
        //std::cerr << '\n';
        o << '\n';
    }
}

// Constructor
cnn_test::cnn_test(xcl_world& world, const char * clFilename, bool isBinary) : m_world(world)
{    
    std::cout << "Initializing OpenCL Kernels (Binary or Source)...\n";
    if(isBinary)
    {
        kernels = get_kernels_binary(world, clFilename, kernel_names); 
    }
    else
    {
        kernels.reserve(kernel_names.size());
        for(std::size_t i = 0; i < kernel_names.size(); ++i)
        {
            kernels.push_back(xcl_import_source(world, clFilename, kernel_names[i].c_str()));
        } 
    }
    std::cout << "OpenCL Kernels Initialized!" << std::endl;
}

// Destructor
cnn_test::~cnn_test()
{
    for(std::size_t i = 0; i < kernels.size(); ++i)
    {
        clReleaseKernel(kernels[i]);
    }
}

//Data cnn_test::seq_img_test(const Data& img)
//{
//    //ModelImporter m_import("lenet_data/model.csv");
//    ModelImporter m_import("../lenet_data/model.csv");
//    std::cout << "Sequential Run Model Data Loaded!\n";
//    auto wc1 = m_import.get_buffer("wc1"); // Conv1 weights
//    auto bc1 = m_import.get_buffer("bc1"); // Conv1 biases
//    auto wc2 = m_import.get_buffer("wc2"); // Conv2 weights
//    auto bc2 = m_import.get_buffer("bc2"); // Conv2 biases
//    auto wd1 = m_import.get_buffer("wd1"); // FullyConn. (dense) weights
//    auto bd1 = m_import.get_buffer("bd1"); // FullyConn. (dense) biases
//    auto wdo = m_import.get_buffer("wdo"); // FullyConn. out (dense) weights
//    auto bdo = m_import.get_buffer("bdo"); // FullyConn. out (dense) biases
//    std::cout << "Sequential Run Model Weights and Biases are Extracted!\n";
//   
//    //std::cerr << std::fixed << std::setprecision(3);
//
//    Data conv1_out = emptyDataBlob<float>({24, 24, 32});
//    Data pool1_out = emptyDataBlob<float>({12, 12, 32});
//    Data conv2_out = emptyDataBlob<float>({8, 8, 64});
//    Data pool2_out = emptyDataBlob<float>({4, 4, 64});
//    Data dens1_out = emptyDataBlob<float>({256});
//    Data class_out = emptyDataBlob<float>({10}); 
//    std::cout << "Intermediate data created!" << std::endl;
//
//    StopWatch<> timer;
//    StopWatch<> conv1_t;
//
//    conv_seq(img.buffer, img.dims.data(),
//             conv1_out.buffer, conv1_out.dims.data(),
//             wc1.buffer, bc1.buffer);
//
//    auto s = 0.0f;
//    for(auto out : conv1_out.buffer)
//    {
//        s += out;
//    }
//    std::cerr << "conv1 out sum: " << s << '\n';
//
//    auto t_conv1 = conv1_t.stop();
//    StopWatch<> pool1_t;
//
//    max_pool2_seq(conv1_out.buffer, conv1_out.dims.data(),
//              pool1_out.buffer, pool1_out.dims.data());
//
//    s = 0.0f;
//    for(auto out : pool1_out.buffer)
//    {
//        s += out;   
//    }
//    std::cerr << "pool1 out sum: " << s << '\n';
//
//    auto t_pool1 = pool1_t.stop();
//    StopWatch<> conv2_t;
//
//    conv_seq(pool1_out.buffer, pool1_out.dims.data(),
//             conv2_out.buffer, conv2_out.dims.data(),
//             wc2.buffer, bc2.buffer);
//
//    s = 0.0f;
//    for(auto out : conv2_out.buffer)
//    {
//        s += out;
//    }
//    std::cerr << "conv2 out sum: " << s << '\n';
//
//    auto t_conv2 = conv2_t.stop();
//    StopWatch<> pool2_t;
//
//    max_pool2_seq(conv2_out.buffer, conv2_out.dims.data(),
//                  pool2_out.buffer, pool2_out.dims.data());
//
//    s = 0.0f;
//    for(auto out : pool2_out.buffer)
//    {
//        s += out;   
//    }
//    std::cerr << "pool2 out sum: " << s << '\n';
//
//    auto t_pool2 = pool2_t.stop();
//    StopWatch<> fc1_t;
//
//    fc_seq(pool2_out.buffer, (pool2_out.dims[0]*pool2_out.dims[1]*pool2_out.dims[2]),
//           dens1_out.buffer, dens1_out.dims[0],
//           wd1.buffer, bd1.buffer);
//    
//    s = 0.0f;
//    for(auto out : dens1_out.buffer)
//    {
//        s += out;
//    }
//    std::cerr << "dens1 out sum: " << s << '\n';
//
//    auto t_fc1 = fc1_t.stop();
//    StopWatch<> fc2_t;
//
//    fc_seq(dens1_out.buffer, dens1_out.dims[0],
//           class_out.buffer, class_out.dims[0],
//           wdo.buffer, bdo.buffer);
//
//    auto t_fc2 = fc2_t.stop();
//    StopWatch<> softmax_t;
//
//    softmax_seq(class_out.buffer, class_out.dims[0], class_out.buffer);
//
//    auto t_soft = softmax_t.stop();
//    auto cpu_elapsed = timer.stop();
//    std::cout << "Total Elapsed Time: " << cpu_elapsed << " us" << '\n';
//    std::cout << "CPU Elapsed Timings: \n";
//    std::cout << "conv1: " << t_conv1 << "\tus"  << '\n';
//    std::cout << "pool1: " << t_pool1 << "\tus"  << '\n';
//    std::cout << "conv2: " << t_conv2 << "\tus"  << '\n';
//    std::cout << "pool2: " << t_pool2 << "\tus"  << '\n';
//    std::cout << "fc1  : " << t_fc1   << "\tus"  << '\n';
//    std::cout << "fc2  : " << t_fc2   << "\tus"  << '\n';
//    std::cout << "soft : " << t_soft  << "\tus"  << '\n';
//    std::cout << std::endl;
//
//    std::size_t class_no = 0;
//    std::cout << std::fixed << std::setprecision(3);
//    for(auto c : class_out.buffer)
//    {
//        std::cout << "Number: " << class_no << "\t\t\t Confidence: %" << c * 100 << '\n';
//        ++class_no;
//    }
//    std::cout << std::scientific << std::setprecision(6) << std::endl;
//
//    return class_out;
//}

//Data cnn_test::ocl_img_test(Data& img)
//{
//    cl_kernel conv = get_kernel_from_vec(kernels, "conv_local");
//    auto conv_reqd_wg = get_kernel_reqd_wg_size(conv, m_world.device_id);
//    cl_kernel maxp = get_kernel_from_vec(kernels, "max_pool2");
//    auto maxp_reqd_wg = get_kernel_reqd_wg_size(maxp, m_world.device_id);
//    cl_kernel fc = get_kernel_from_vec(kernels, "fc");
//    auto fc_reqd_wg = get_kernel_reqd_wg_size(fc, m_world.device_id);
//    cl_kernel softmax = get_kernel_from_vec(kernels, "softmax_layer");
//    auto softmax_reqd_wg = get_kernel_reqd_wg_size(softmax, m_world.device_id);
//
//    auto cl_img = data_host_to_device(m_world, CL_MEM_READ_ONLY, img);
//
//    //ModelImporter m_import("lenet_data/model.csv");
//    ModelImporter m_import("../lenet_data/model.csv");
//    std::cout << "OpenCL Run Model Data Loaded!\n";
//    auto wc1 = m_import.get_buffer("wc1"); // Conv1 weights
//    auto bc1 = m_import.get_buffer("bc1"); // Conv1 biases
//    auto wc2 = m_import.get_buffer("wc2"); // Conv2 weights
//    auto bc2 = m_import.get_buffer("bc2"); // Conv2 biases
//    auto wd1 = m_import.get_buffer("wd1"); // FullyConn. (dense) weights
//    auto bd1 = m_import.get_buffer("bd1"); // FullyConn. (dense) biases
//    auto wdo = m_import.get_buffer("wdo"); // FullyConn. out (dense) weights
//    auto bdo = m_import.get_buffer("bdo"); // FullyConn. out (dense) biases
//     
//    auto cl_wc1 = data_host_to_device(m_world, CL_MEM_READ_ONLY, wc1);
//    auto cl_bc1 = data_host_to_device(m_world, CL_MEM_READ_ONLY, bc1);
//    auto cl_wc2 = data_host_to_device(m_world, CL_MEM_READ_ONLY, wc2);
//    auto cl_bc2 = data_host_to_device(m_world, CL_MEM_READ_ONLY, bc2);
//    auto cl_wd1 = data_host_to_device(m_world, CL_MEM_READ_ONLY, wd1);
//    auto cl_bd1 = data_host_to_device(m_world, CL_MEM_READ_ONLY, bd1);
//    auto cl_wdo = data_host_to_device(m_world, CL_MEM_READ_ONLY, wdo);
//    auto cl_bdo = data_host_to_device(m_world, CL_MEM_READ_ONLY, bdo);
//    std::cout << "OpenCL Run Model Weights and Biases are Extracted!\n";
//
//    auto conv1_out = emptyClDataBlob<float>(m_world, {24, 24, 32}, CL_MEM_READ_WRITE);
//    auto pool1_out = emptyClDataBlob<float>(m_world, {12, 12, 32}, CL_MEM_READ_WRITE);
//    auto conv2_out = emptyClDataBlob<float>(m_world, {8, 8, 64}, CL_MEM_READ_WRITE);
//    auto pool2_out = emptyClDataBlob<float>(m_world, {4, 4, 64}, CL_MEM_READ_WRITE);
//    auto dens1_out = emptyClDataBlob<float>(m_world, {256}, CL_MEM_READ_WRITE);
//    auto class_out = emptyClDataBlob<float>(m_world, {10}, CL_MEM_READ_WRITE);
//    auto softm_out = emptyClDataBlob<float>(m_world, {10}, CL_MEM_WRITE_ONLY);
//    std::cout << "Intermediate data created!" << std::endl;
//
//    cl_uchar conv1_in_width   = static_cast<cl_uchar>(cl_img.dims[0]);
//    cl_uchar conv1_in_height  = static_cast<cl_uchar>(cl_img.dims[1]);
//    cl_uchar conv1_mask_depth = static_cast<cl_uchar>(cl_img.dims[2]);
//    cl_uchar conv2_in_width   = static_cast<cl_uchar>(pool1_out.dims[0]);
//    cl_uchar conv2_in_height  = static_cast<cl_uchar>(pool1_out.dims[1]);
//    cl_uchar conv2_mask_depth = static_cast<cl_uchar>(pool1_out.dims[2]);
//
//    clSetKernelArg(conv, 0, sizeof(cl_mem), &cl_img.buffer); 
//    clSetKernelArg(conv, 1, sizeof(cl_mem), &conv1_out.buffer);
//    clSetKernelArg(conv, 2, sizeof(cl_mem), &cl_wc1.buffer);
//    clSetKernelArg(conv, 3, sizeof(cl_mem), &cl_bc1.buffer);
//    clSetKernelArg(conv, 4, sizeof(cl_uchar), &conv1_in_width);
//    clSetKernelArg(conv, 5, sizeof(cl_uchar), &conv1_in_height);
//    clSetKernelArg(conv, 6, sizeof(cl_uchar), &conv1_mask_depth);
//
//    StopWatch<> timer;
//
//    size_t global[3] = {24, 24, 32};
//    auto t_conv1 = launch_kernel(m_world, conv, global, conv_reqd_wg.data());
//
//    clSetKernelArg(maxp, 0, sizeof(cl_mem), &conv1_out.buffer);
//    clSetKernelArg(maxp, 1, sizeof(cl_mem), &pool1_out.buffer);
//
//    global[0] = 12; global[1] = 12; global[2] = 32;
//    auto t_pool1 = launch_kernel(m_world, maxp, global, maxp_reqd_wg.data());
//
//    clSetKernelArg(conv, 0, sizeof(cl_mem), &pool1_out.buffer);
//    clSetKernelArg(conv, 1, sizeof(cl_mem), &conv2_out.buffer);
//    clSetKernelArg(conv, 2, sizeof(cl_mem), &cl_wc2.buffer);
//    clSetKernelArg(conv, 3, sizeof(cl_mem), &cl_bc2.buffer);
//    clSetKernelArg(conv, 4, sizeof(cl_uchar), &conv2_in_width);
//    clSetKernelArg(conv, 5, sizeof(cl_uchar), &conv2_in_height);
//    clSetKernelArg(conv, 6, sizeof(cl_uchar), &conv2_mask_depth);
//
//    global[0] = 8; global[1] = 8; global[2] = 64;
//    auto t_conv2 = launch_kernel(m_world, conv, global, conv_reqd_wg.data());
//
//    clSetKernelArg(maxp, 0, sizeof(cl_mem), &conv2_out.buffer);
//    clSetKernelArg(maxp, 1, sizeof(cl_mem), &pool2_out.buffer);
//
//    global[0] = 4; global[1] = 4; global[2] = 64;
//    auto t_pool2 = launch_kernel(m_world, maxp, global, maxp_reqd_wg.data());
//
//    cl_ushort in_neuron1 = static_cast<cl_ushort>(4 * 4 * 64);
//
//    clSetKernelArg(fc, 0, sizeof(cl_mem), &pool2_out.buffer);
//    clSetKernelArg(fc, 1, sizeof(cl_mem), &dens1_out.buffer);
//    clSetKernelArg(fc, 2, sizeof(cl_mem), &cl_wd1.buffer);
//    clSetKernelArg(fc, 3, sizeof(cl_mem), &cl_bd1.buffer);
//    clSetKernelArg(fc, 4, sizeof(cl_ushort), &in_neuron1);
//
//    global[0] = 256; global[1] = 1; global[2] = 1;
//    auto t_fc1 = launch_kernel(m_world, fc, global, fc_reqd_wg.data());
//
//    cl_ushort in_neuron2 = static_cast<cl_ushort>(256);
//
//    clSetKernelArg(fc, 0, sizeof(cl_mem), &dens1_out.buffer);
//    clSetKernelArg(fc, 1, sizeof(cl_mem), &class_out.buffer);
//    clSetKernelArg(fc, 2, sizeof(cl_mem), &cl_wdo.buffer);
//    clSetKernelArg(fc, 3, sizeof(cl_mem), &cl_bdo.buffer);
//    clSetKernelArg(fc, 4, sizeof(cl_ushort), &in_neuron2);
//
//    global[0] = 10; global[1] = 1; global[2] = 1;
//    auto t_fc2 = launch_kernel(m_world, fc, global, fc_reqd_wg.data());
//
//    clSetKernelArg(softmax, 0, sizeof(cl_mem), &class_out.buffer);
//    clSetKernelArg(softmax, 1, sizeof(cl_mem), &softm_out.buffer);
//
//    global[0] = 1; global[1] = 1; global[2] = 1;
//    auto t_soft = launch_kernel(m_world, softmax, global, softmax_reqd_wg.data());
//
//    auto fpga_elapsed = timer.stop();
//
//    auto cnn_outs = data_device_to_host(m_world, softm_out);
//
//    std::cout << "Total Time Elapsed: " << (std::size_t)fpga_elapsed << "\tus"  << '\n';
//    std::cout << "FPGA Elapsed Timings: \n";
//    std::cout << "conv1: " << (std::size_t)t_conv1 / 1000 << "\tus"  << '\n';
//    std::cout << "pool1: " << (std::size_t)t_pool1 / 1000 << "\tus"  << '\n';
//    std::cout << "conv2: " << (std::size_t)t_conv2 / 1000 << "\tus"  << '\n';
//    std::cout << "pool2: " << (std::size_t)t_pool2 / 1000 << "\tus"  << '\n';
//    std::cout << "fc1  : " << (std::size_t)t_fc1   / 1000 << "\tus"  << '\n';
//    std::cout << "fc2  : " << (std::size_t)t_fc2   / 1000 << "\tus"  << '\n';
//    std::cout << "soft : " << (std::size_t)t_soft  / 1000 << "\tus"  << '\n';
//    std::cout << std::endl;
//
//    std::size_t class_no = 0;
//    std::cout << std::fixed << std::setprecision(3);
//    for(auto c : cnn_outs.buffer)
//    {
//        std::cout << "Number: " << class_no << "\t\t\t Confidence: %" << c * 100 << '\n';
//        ++class_no;
//    }
//
//    std::cout << std::scientific << std::setprecision(6) << std::endl;
//
//    clReleaseMemObject(cl_img.buffer);
//    clReleaseMemObject(cl_wc1.buffer);
//    clReleaseMemObject(cl_bc1.buffer);
//    clReleaseMemObject(cl_wc2.buffer);
//    clReleaseMemObject(cl_bc2.buffer);
//    clReleaseMemObject(cl_wd1.buffer);
//    clReleaseMemObject(cl_bd1.buffer);
//    clReleaseMemObject(cl_wdo.buffer);
//    clReleaseMemObject(cl_bdo.buffer);
//    return cnn_outs;
//}


float cnn_test::test_img()
{
    Data img = getTestImg();
    //auto img_data = mnist_test::get_a_img("lenet_data/test_img1_7");
    //Data img;
    //img.dims = {28, 28, 1};
    //img.buffer = img_data;

    std::cout << "test image generated!" << std::endl;

    ModelImporter imp("../lenet_data/model.csv");
    auto model = imp.get_buffers();
    
    auto seq_out = seq_cnn_img_test(img, model);
    auto ocl_out = ocl_cnn_img_test(img, model, m_world, kernels);

//    auto seq_out = seq_img_test(img);
//    auto ocl_out = ocl_img_test(img);

    if(seq_out.buffer.size() != ocl_out.buffer.size())
    {
        std::cerr << "test_img output buffer sizes are not equal!!\n";
        return 0.0f;
    }
    else
    {
        return absolute(seq_out.buffer, ocl_out.buffer);   
    }
}
