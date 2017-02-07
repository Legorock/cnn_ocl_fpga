#include "cnn_test.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "genData.h"
#include "seq.h"

#include "ModelImporter.h"
#include "mnist_test_img.h"

#include "helper.h"
#include "Measure.h"

template<typename T>
inline static void print_buf(std::ostream& o, const T *  buf, 
        const std::vector<std::size_t>& dims, const std::size_t curr_dim);

//const std::vector<std::string> kernel_names = {"max_pool2", "conv_local_flatmem", "softmax_layer", "fully_connected_local"};
//const std::vector<std::string> kernel_names = {"max_pool2", "conv_layer", "softmax_layer", "fully_connected_local"};
//const std::vector<std::string> kernel_names = {"max_pool2", "conv_local_flatasync", "softmax_layer", "fully_connected_local"};
//const std::vector<std::string> kernel_names = {"max_pool2", "conv_local", "softmax_layer", "fc_local"};
const std::vector<std::string> kernel_names = {"max_pool2", "conv_local", "softmax_layer", "fc"};
const std::vector<std::string> kernel_layers = {"max_pool", "conv", "fc", "softmax"};


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

//    template<typename diff_method>
//inline static float diff_data(const krnl_data_map & seq, const krnl_data_map & ocl, diff_method method)
//{
//    float diff = 0.0f;
//    for(auto & layer_name : kernel_layers)
//    {
//        const auto & seq_buffer = seq.at(layer_name).buffer;
//        const auto & ocl_buffer = ocl.at(layer_name).buffer;
//        
//        if(seq_buffer.size() != ocl_buffer.size())
//        {
//            std::cerr << "diff_data: error buffer sizes doesn't match on layer " << layer_name << ")\n";
//            continue;
//        }
//        std::cerr << "Kernel layer: " << layer_name << '\t' << diff << '\n';
//        diff += method(seq_buffer, ocl_buffer);
//    }
//    return diff;
//}

inline static Data getTestImg()
{
    Data im = emptyDataBlob<float>({28, 28, 1});
    
    for(std::size_t i = 0; i < 28 * 28; ++i)
    {
        im.buffer[i] = static_cast<float>(mnist_test::img[i]) / 255;
    }
    return im;
}

//inline static krnl_data_map sequential_runs(std::map<std::string, std::vector<Data>>& data)
//{
//    krnl_data_map outs;
//
//    for(auto & krnl_run : data)
//    {
//        const std::string & krnl_name = krnl_run.first;
//        std::vector<Data> & krnl_data = krnl_run.second;
//        if(krnl_name == "max_pool")
//        {
//            //std::cerr << "maxpool sequential test!\n";
//            auto v_in_dims = krnl_data[0].dims;
//            auto v_out_dims = krnl_data[1].dims;
//            std::size_t in_dims[3] = {v_in_dims[0], v_in_dims[1], v_in_dims[2]};
//            std::size_t out_dims[3] = {v_out_dims[0], v_out_dims[1], v_out_dims[2]};
//            max_pool2_seq(krnl_data[0].buffer, in_dims, krnl_data[1].buffer, out_dims);
//            //std::cerr << "maxpool sequential ends!\n";
//        }
//        else if(krnl_name == "conv")
//        {
//            //std::cerr << "conv sequential test!\n";
//            auto v_in_dims = krnl_data[0].dims;
//            auto v_out_dims = krnl_data[1].dims;
//            std::size_t in_dims[3] = {v_in_dims[0], v_in_dims[1], v_in_dims[2]};
//            std::size_t out_dims[3] = {v_out_dims[0], v_out_dims[1], v_out_dims[2]};
//            conv_seq(krnl_data[0].buffer, in_dims, krnl_data[1].buffer, out_dims, krnl_data[2].buffer, krnl_data[3].buffer);
//            //std::cerr << "conv sequnetial ends!\n";
//        }
//        else if(krnl_name == "softmax")
//        {
//            //std::cerr << "softmax sequential test!\n";
//            std::size_t in_dim = krnl_data[0].dims[0];
//            softmax_seq(krnl_data[0].buffer, in_dim, krnl_data[1].buffer);
//            //std::cerr << "softmax sequential ends!\n";
//        }
//        else if(krnl_name == "fc")
//        {
//            //std::cerr << "fc sequential test!\n";
//            std::size_t in_dim = krnl_data[0].dims[0];
//            std::size_t out_dim = krnl_data[1].dims[0];
//            fc_seq(krnl_data[0].buffer, in_dim, krnl_data[1].buffer, out_dim, krnl_data[2].buffer, krnl_data[3].buffer);
//            //std::cerr << "fc sequential ends!\n";
//        }
//
//        std::cout << krnl_name << " sequential tested!" << std::endl;
//        outs[krnl_name] = krnl_data[1];
//    }
//    return outs;
//}
//
//inline static std::map<std::string, std::vector<Data>> gen_run_data()
//{
//    std::map<std::string, std::vector<Data>> data;
//    for(auto & krnl_name : kernel_layers)
//    {
//        if(krnl_name == "max_pool")
//        {
//            Data in, out;
//            in.dims = {24, 24, 4};
//            out.dims = {in.dims[0]/2, in.dims[1]/2, in.dims[2]};
//            in.buffer = gen3Data<'d'>(in.dims[0], in.dims[1], in.dims[2]);
//            out.buffer.resize(out.dims[0] * out.dims[1] * out.dims[2], 0.0f);
//
//            data[krnl_name] = std::vector<Data>({in, out});
//        }
//        else if(krnl_name == "conv")
//        {
//            Data in, out, w, b;
//            in.dims = {12, 12, 2};
//            //out.dims = {in.dims[0]-4, in.dims[1]-4, 32};
//            out.dims = {in.dims[0]-4, in.dims[1]-4, 4};
//            w.dims = {5 * 5 * in.dims[2] * out.dims[2]};
//            b.dims = {out.dims[2]};
//            in.buffer = gen3Data<'w'>(in.dims[0], in.dims[1], in.dims[2]);
//            out.buffer.resize(out.dims[0] * out.dims[1] * out.dims[2], 0.0f);
//            w.buffer = gen1Data(w.dims[0]);
//            b.buffer = gen1Data(b.dims[0]);
//
//            data[krnl_name] = std::vector<Data>({in, out, w, b});
//        }
//        else if(krnl_name == "softmax")
//        {
//            Data in, out;
//            in.dims = {10};
//            out.dims = {in.dims};
//            in.buffer = gen1Data(in.dims[0]);
//            out.buffer.resize(out.dims[0], 0.0f);
//
//            data[krnl_name] = std::vector<Data>({in, out});
//        }
//        else if(krnl_name == "fc")
//        {
//            Data in, out, w, b;
//            in.dims = {512};
//            out.dims = {256};
//            w.dims = {in.dims[0] * out.dims[0]};
//            b.dims = {out.dims[0]};
//            in.buffer = gen1Data(in.dims[0]);
//            out.buffer.resize(out.dims[0], 0.0f);
//            w.buffer = std::vector<float>(w.dims[0], 1.0f);
//            b.buffer = gen1Data(b.dims[0]);
//
//            data[krnl_name] = std::vector<Data>({in, out, w, b});
//        }
//    }
//    return data;
//}

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
inline static void print_test_results(std::ofstream & file_out, 
        const krnl_data_map & out_data,
        const std::string & out_kernel_name = std::string("all"))
{
    std::cout << "printing test result!" << std::endl;
    for(const auto & d : out_data)
    {
        auto & kernel_name = d.first;
        if(out_kernel_name != "all" && out_kernel_name != kernel_name)
            continue;

        auto & out = d.second; 

        file_out << kernel_name << " results!\n";
        auto & dims = out.dims;
        auto & buff = out.buffer;

        print_buf<float>(file_out, buff.data(), dims, dims.size()-1);
        file_out << std::endl;
    }
    file_out << std::endl;
}

//void cnn_test::ocl_maxpool2_run(cl_kernel kernel, std::vector<Data>& data)
//{
//    auto & data_in = data[0];
//    auto & data_out = data[1];
//    size_t buf_in_size = data_in.dims[0] * data_in.dims[1] * data_in.dims[2];
//    size_t buf_out_size = data_out.dims[0] * data_out.dims[1] * data_out.dims[2];
//
//    buf_in_size *= sizeof(float);
//    buf_out_size *= sizeof(float);
//
//    // Device buffer allocation
//    cl_mem buf_in = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_in_size);
//    cl_mem buf_out = xcl_malloc(test_world, CL_MEM_WRITE_ONLY, buf_out_size);
//
//    // HOST to DEVICE transfer
//    xcl_memcpy_to_device(test_world, buf_in, data_in.buffer.data(), buf_in_size);
//
//    cl_int err = CL_SUCCESS;
//    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
//    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
//    if(err != CL_SUCCESS)
//    {
//        std::cerr << "One of the clSetKernelArg failed!\n";
//        std::exit(EXIT_FAILURE);
//    }
//
//    auto reqd_wg_size = get_kernel_reqd_wg_size(kernel, test_world.device_id);
//
//    size_t global[3] = {data_out.dims[0], data_out.dims[1], data_out.dims[2]};
//    size_t local[3] = {reqd_wg_size[0], reqd_wg_size[1], reqd_wg_size[2]};
//    std::cout << "global: " << global[0] << '\t' << global[1] << '\t' << global[2] << '\n';
//    std::cout << "local: " << local[0] << '\t' << local[1] << '\t' << local[2] << std::endl;
//
//    std::cout << "OpenCL kernel launching!" << std::endl;
//    auto exe_t = launch_kernel(test_world, kernel, global, local); 
//    std::cout << "OpenCL kernel finished!" << std::endl;
//
//    std::cout << "Max Pooling kernel exe time(ns): " << exe_t << " ns" << std::endl; 
//
//    // DEVICE to HOST transfer
//    xcl_memcpy_from_device(test_world, data_out.buffer.data(), buf_out, buf_out_size);
//
//    clReleaseMemObject(buf_in);
//    clReleaseMemObject(buf_out);
//}
//
//void cnn_test::ocl_conv_run(cl_kernel kernel, std::vector<Data>& data)
//{
//    auto & data_in = data[0];
//    auto & data_out = data[1];
//    auto & data_weight = data[2];
//    auto & data_biases = data[3];
//    auto & in_dims = data[0].dims;
//    auto & out_dims = data[1].dims;
//    size_t buf_in_size = in_dims[0] * in_dims[1] * in_dims[2]  * sizeof(float);
//    size_t buf_out_size = out_dims[0] * out_dims[1] * out_dims[2] * sizeof(float);
//    size_t buf_w_size = data[2].dims[0] * sizeof(float);
//    size_t buf_b_size = data[3].dims[0] * sizeof(float);
//
//    // Device buffer allocation
//    cl_mem buf_in = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_in_size);
//    cl_mem buf_out = xcl_malloc(test_world, CL_MEM_WRITE_ONLY, buf_out_size);
//    cl_mem buf_w = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_w_size);
//    cl_mem buf_b = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_b_size);
//
//    // HOST to DEVICE transfer
//    xcl_memcpy_to_device(test_world, buf_in, data_in.buffer.data(), buf_in_size);
//    xcl_memcpy_to_device(test_world, buf_w, data_weight.buffer.data(), buf_w_size);
//    xcl_memcpy_to_device(test_world, buf_b, data_biases.buffer.data(), buf_b_size);
//
//    cl_uchar in_width = static_cast<cl_uchar>(in_dims[0]);
//    cl_uchar in_height = static_cast<cl_uchar>(in_dims[1]);
//    cl_uchar mask_depth = static_cast<cl_uchar>(in_dims[2]);
//
//    cl_int err = CL_SUCCESS;
//    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
//    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
//    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_w);
//    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_b);
//    err |= clSetKernelArg(kernel, 4, sizeof(cl_uchar), &in_width);
//    err |= clSetKernelArg(kernel, 5, sizeof(cl_uchar), &in_height);
//    err |= clSetKernelArg(kernel, 6, sizeof(cl_uchar), &mask_depth);
//
//    if(err != CL_SUCCESS)
//    {
//        std::cerr << "One of the clSetKernelArg failed!\n";
//        std::exit(EXIT_FAILURE);
//    }
//
//    auto reqd_wg_size = get_kernel_reqd_wg_size(kernel, test_world.device_id);
//    if((reqd_wg_size[0]*reqd_wg_size[1]*reqd_wg_size[2]) == 0)
//    {
//        reqd_wg_size = {4, 4, 2};
//    }
//
//    size_t global[3] = {out_dims[0], out_dims[1], out_dims[2]};
//    size_t local[3] = {reqd_wg_size[0], reqd_wg_size[1], reqd_wg_size[2]};
//    std::cout << "global: " << global[0] << '\t' << global[1] << '\t' << global[2] << '\n';
//    std::cout << "local: " << local[0] << '\t' << local[1] << '\t' << local[2] << std::endl;
//
//    std::cout << "OpenCL kernel launching!" << std::endl;
//    auto exe_t = launch_kernel(test_world, kernel, global, local); 
//    std::cout << "OpenCL kernel finished!" << std::endl;
//
//    std::cout << "Convolution kernel exe time(ns): " << exe_t << " ns" << std::endl; 
//
//    // DEVICE to HOST transfer
//    xcl_memcpy_from_device(test_world, data_out.buffer.data(), buf_out, buf_out_size);
//    
//    clReleaseMemObject(buf_in);
//    clReleaseMemObject(buf_out);
//    clReleaseMemObject(buf_w);
//    clReleaseMemObject(buf_b);
//}
//
//void cnn_test::ocl_fc_run(cl_kernel kernel, std::vector<Data>& data)
//{
//    auto & data_in = data[0];
//    auto & data_out = data[1];
//    auto & data_weight = data[2];
//    auto & data_biases = data[3];
//    size_t buf_in_size = data[0].dims[0] * sizeof(float);
//    size_t buf_out_size = data[1].dims[0] * sizeof(float);
//    size_t buf_w_size = data[2].dims[0] * sizeof(float);
//    size_t buf_b_size = data[3].dims[0] * sizeof(float);
//
//    // Device buffer allocation
//    cl_mem buf_in = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_in_size);
//    cl_mem buf_out = xcl_malloc(test_world, CL_MEM_WRITE_ONLY, buf_out_size);
//    cl_mem buf_w = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_w_size);
//    cl_mem buf_b = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_b_size);
//
//    // HOST to DEVICE transfer
//    xcl_memcpy_to_device(test_world, buf_in, data_in.buffer.data(), buf_in_size);
//    xcl_memcpy_to_device(test_world, buf_w, data_weight.buffer.data(), buf_w_size);
//    xcl_memcpy_to_device(test_world, buf_b, data_biases.buffer.data(), buf_b_size);
//
//    cl_ushort in_neuron = static_cast<cl_ushort>(data[0].dims[0]);
//
//    cl_int err = CL_SUCCESS;
//    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
//    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
//    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_w);
//    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_b);
//    err |= clSetKernelArg(kernel, 4, sizeof(cl_ushort), &in_neuron); 
//    if(err != CL_SUCCESS)
//    {
//        std::cerr << "One of the clSetKernelArg failed!\n";
//        std::exit(EXIT_FAILURE);
//    }
//
//    auto reqd_wg_size = get_kernel_reqd_wg_size(kernel, test_world.device_id);
//
//    size_t global[3] = {data[1].dims[0], 1, 1};
//    size_t local[3] = {reqd_wg_size[0], reqd_wg_size[1], reqd_wg_size[2]};
//    std::cout << "global: " << global[0] << '\t' << global[1] << '\t' << global[2] << '\n';
//    std::cout << "local: " << local[0] << '\t' << local[1] << '\t' << local[2] << std::endl;
//
//    std::cout << "OpenCL kernel launching!" << std::endl;
//    auto exe_t = launch_kernel(test_world, kernel, global, local); 
//    std::cout << "OpenCL kernel finished!" << std::endl;
//
//    std::cout << "FullyConnected kernel exe time(ns): " << exe_t << " ns" << std::endl; 
//
//    // DEVICE to HOST transfer
//    xcl_memcpy_from_device(test_world, data_out.buffer.data(), buf_out, buf_out_size);
//
//    clReleaseMemObject(buf_in);
//    clReleaseMemObject(buf_out);
//    clReleaseMemObject(buf_w);
//    clReleaseMemObject(buf_b);
//}
//
//void cnn_test::ocl_softmax_run(cl_kernel kernel, std::vector<Data>& data)
//{
//    auto & data_in = data[0];
//    auto & data_out = data[1];
//    size_t buf_in_size = data[0].dims[0] * sizeof(float);
//    size_t buf_out_size = data[1].dims[0] * sizeof(float);
//
//    // Device buffer allocation
//    cl_mem buf_in = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_in_size);
//    cl_mem buf_out = xcl_malloc(test_world, CL_MEM_WRITE_ONLY, buf_out_size);
//
//    // HOST to DEVICE transfer
//    xcl_memcpy_to_device(test_world, buf_in, data_in.buffer.data(), buf_in_size);
//
//    cl_int err = CL_SUCCESS;
//    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
//    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
//    if(err != CL_SUCCESS)
//    {
//        std::cerr << "One of the clSetKernelArg failed!\n";
//        std::exit(EXIT_FAILURE);
//    }
//
//    size_t global[3] = {1,1,1};
//    size_t local[3] = {1,1,1};
//    std::cout << "global: " << global[0] << '\t' << global[1] << '\t' << global[2] << '\n';
//    std::cout << "local: " << local[0] << '\t' << local[1] << '\t' << local[2] << std::endl;
//
//    std::cout << "OpenCL kernel launching!" << std::endl;
//    auto exe_t = launch_kernel(test_world, kernel, global, local); 
//    std::cout << "OpenCL kernel finished!" << std::endl;
//
//    std::cout << "Softmax kernel exe time(ns): " << exe_t << " ns" << std::endl; 
//
//    // DEVICE to HOST transfer
//    xcl_memcpy_from_device(test_world, data_out.buffer.data(), buf_out, buf_out_size);
//
//    clReleaseMemObject(buf_in);
//    clReleaseMemObject(buf_out);
//}
//
//krnl_data_map cnn_test::ocl_runs(std::map<std::string, std::vector<Data>>& data)
//{
//    krnl_data_map out;
//
//    for(cl_kernel kernel : kernels)
//    {
//        auto kernel_name = get_kernel_name(kernel);
//        std::cout << "actual ocl kernel name: " << kernel_name << std::endl;
//        if(kernel_name.find("max_pool") != std::string::npos)
//        {
//            ocl_maxpool2_run(kernel, data["max_pool"]);
//            out["max_pool"] = data["max_pool"][1];
//        }
//        else if(kernel_name.find("conv") != std::string::npos)
//        {
//            ocl_conv_run(kernel, data["conv"]);
//            out["conv"] = data["conv"][1];
//        }
//        else if(kernel_name.find("fc") != std::string::npos)
//        {
//            ocl_fc_run(kernel, data["fc"]);
//            out["fc"] = data["fc"][1];
//        }
//        else if(kernel_name.find("softmax") != std::string::npos)
//        {
//            ocl_softmax_run(kernel, data["softmax"]);
//            out["softmax"] = data["softmax"][1];
//        }
//        else
//        {
//            std::cerr << "kernel_name does not match any possibility of layer!\n";
//            continue;
//        }
//    }
//    return out;
//}

// Constructor
cnn_test::cnn_test(xcl_world& world, const char * clFilename, bool isBinary) : test_world(world)
{    
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
}

// Destructor
cnn_test::~cnn_test()
{
    for(std::size_t i = 0; i < kernels.size(); ++i)
    {
        clReleaseKernel(kernels[i]);
    }
}

//float cnn_test::test()
//{
//    auto seq_data = gen_run_data();
//    std::cout << "data generation for test ends!" << std::endl;
//    // Sequential CPU run to form a valid output values
//    std::cout << "Sequential Runs start!" << std::endl;
//    auto seq_out = sequential_runs(seq_data);
//    std::cout << "Sequential Runs end!" << std::endl;
//    std::ofstream seq_test_out("seq_out.txt");
//    print_test_results(seq_test_out, seq_out, "all");
//    seq_test_out.close();
//
//    auto ocl_data = gen_run_data();
//    std::cout << "data generation for test ends!" << std::endl;
//    // OpenCL test run
//    std::cout << "OpenCL Runs start!" << std::endl;
//    auto ocl_out = ocl_runs(ocl_data);
//    std::cout << "OpenCL Runs end!" << std::endl;
//    std::ofstream ocl_test_out("ocl_out.txt");
//    print_test_results(ocl_test_out, ocl_out, "all");
//    ocl_test_out.close();
//    
////    return 0.0f;
//    return diff_data(seq_out, ocl_out, absolute);
//}

Data cnn_test::seq_img_test(const Data& img)
{
    ModelImporter m_import("lenet_data/model.csv");
    std::cout << "Sequential Run Model Data Loaded!\n";
    auto wc1 = m_import.get_buffer("wc1"); // Conv1 weights
    auto bc1 = m_import.get_buffer("bc1"); // Conv1 biases
    auto wc2 = m_import.get_buffer("wc2"); // Conv2 weights
    auto bc2 = m_import.get_buffer("bc2"); // Conv2 biases
    auto wd1 = m_import.get_buffer("wd1"); // FullyConn. (dense) weights
    auto bd1 = m_import.get_buffer("bd1"); // FullyConn. (dense) biases
    auto wdo = m_import.get_buffer("wdo"); // FullyConn. out (dense) weights
    auto bdo = m_import.get_buffer("bdo"); // FullyConn. out (dense) biases
    std::cout << "Sequential Run Model Weights and Biases are Extracted!\n";
   
    //std::cerr << std::fixed << std::setprecision(3);

    Data conv1_out = emptyDataBlob<float>({24, 24, 32});
    Data pool1_out = emptyDataBlob<float>({12, 12, 32});
    Data conv2_out = emptyDataBlob<float>({8, 8, 64});
    Data pool2_out = emptyDataBlob<float>({4, 4, 64});
    Data dens1_out = emptyDataBlob<float>({256});
    Data class_out = emptyDataBlob<float>({10}); 
    std::cout << "Intermediate data created!" << std::endl;

    StopWatch<> timer;
    StopWatch<> conv1_t;

    conv_seq(img.buffer, img.dims.data(),
             conv1_out.buffer, conv1_out.dims.data(),
             wc1.buffer, bc1.buffer);

    auto s = 0.0f;
    for(auto out : conv1_out.buffer)
    {
        s += out;
    }
    std::cerr << "conv1 out sum: " << s << '\n';

    auto t_conv1 = conv1_t.stop();
    StopWatch<> pool1_t;

    max_pool2_seq(conv1_out.buffer, conv1_out.dims.data(),
              pool1_out.buffer, pool1_out.dims.data());

    s = 0.0f;
    for(auto out : pool1_out.buffer)
    {
        s += out;   
    }
    std::cerr << "pool1 out sum: " << s << '\n';

    auto t_pool1 = pool1_t.stop();
    StopWatch<> conv2_t;

    conv_seq(pool1_out.buffer, pool1_out.dims.data(),
             conv2_out.buffer, conv2_out.dims.data(),
             wc2.buffer, bc2.buffer);

    s = 0.0f;
    for(auto out : conv2_out.buffer)
    {
        s += out;
    }
    std::cerr << "conv2 out sum: " << s << '\n';

    auto t_conv2 = conv2_t.stop();
    StopWatch<> pool2_t;

    max_pool2_seq(conv2_out.buffer, conv2_out.dims.data(),
                  pool2_out.buffer, pool2_out.dims.data());

    s = 0.0f;
    for(auto out : pool2_out.buffer)
    {
        s += out;   
    }
    std::cerr << "pool2 out sum: " << s << '\n';

    auto t_pool2 = pool2_t.stop();
    StopWatch<> fc1_t;

    fc_seq(pool2_out.buffer, (pool2_out.dims[0]*pool2_out.dims[1]*pool2_out.dims[2]),
           dens1_out.buffer, dens1_out.dims[0],
           wd1.buffer, bd1.buffer);
    
    s = 0.0f;
    for(auto out : dens1_out.buffer)
    {
        s += out;
    }
    std::cerr << "dens1 out sum: " << s << '\n';

    auto t_fc1 = fc1_t.stop();
    StopWatch<> fc2_t;

    fc_seq(dens1_out.buffer, dens1_out.dims[0],
           class_out.buffer, class_out.dims[0],
           wdo.buffer, bdo.buffer);

    auto t_fc2 = fc2_t.stop();
    StopWatch<> softmax_t;

    softmax_seq(class_out.buffer, class_out.dims[0], class_out.buffer);

    auto t_soft = softmax_t.stop();
    auto cpu_elapsed = timer.stop();
    std::cout << "Total Elapsed Time: " << cpu_elapsed << " us" << '\n';
    std::cout << "CPU Elapsed Timings: \n";
    std::cout << "conv1: " << t_conv1 << "\tus"  << '\n';
    std::cout << "pool1: " << t_pool1 << "\tus"  << '\n';
    std::cout << "conv2: " << t_conv2 << "\tus"  << '\n';
    std::cout << "pool2: " << t_pool2 << "\tus"  << '\n';
    std::cout << "fc1  : " << t_fc1   << "\tus"  << '\n';
    std::cout << "fc2  : " << t_fc2   << "\tus"  << '\n';
    std::cout << "soft : " << t_soft  << "\tus"  << '\n';
    std::cout << std::endl;

    std::size_t class_no = 0;
    std::cout << std::fixed << std::setprecision(3);
    for(auto c : class_out.buffer)
    {
        std::cout << "Number: " << class_no << "\t\t\t Confidence: %" << c * 100 << '\n';
        ++class_no;
    }
    std::cout << std::scientific << std::setprecision(6) << std::endl;

    return class_out;
//    return img; // Placeholder
}

Data cnn_test::ocl_img_test(Data& img)
{
    cl_kernel conv = get_kernel_from_vec(kernels, "conv_local");
    auto conv_reqd_wg = get_kernel_reqd_wg_size(conv, test_world.device_id);
    cl_kernel maxp = get_kernel_from_vec(kernels, "max_pool2");
    auto maxp_reqd_wg = get_kernel_reqd_wg_size(maxp, test_world.device_id);
    cl_kernel fc = get_kernel_from_vec(kernels, "fc");
    auto fc_reqd_wg = get_kernel_reqd_wg_size(fc, test_world.device_id);
    cl_kernel softmax = get_kernel_from_vec(kernels, "softmax_layer");
    auto softmax_reqd_wg = get_kernel_reqd_wg_size(softmax, test_world.device_id);

    auto cl_img = data_host_to_device(test_world, CL_MEM_READ_ONLY, img);

    ModelImporter m_import("lenet_data/model.csv");
    std::cout << "OpenCL Run Model Data Loaded!\n";
    auto wc1 = m_import.get_buffer("wc1"); // Conv1 weights
    auto bc1 = m_import.get_buffer("bc1"); // Conv1 biases
    auto wc2 = m_import.get_buffer("wc2"); // Conv2 weights
    auto bc2 = m_import.get_buffer("bc2"); // Conv2 biases
    auto wd1 = m_import.get_buffer("wd1"); // FullyConn. (dense) weights
    auto bd1 = m_import.get_buffer("bd1"); // FullyConn. (dense) biases
    auto wdo = m_import.get_buffer("wdo"); // FullyConn. out (dense) weights
    auto bdo = m_import.get_buffer("bdo"); // FullyConn. out (dense) biases
     
    auto cl_wc1 = data_host_to_device(test_world, CL_MEM_READ_ONLY, wc1);
    auto cl_bc1 = data_host_to_device(test_world, CL_MEM_READ_ONLY, bc1);
    auto cl_wc2 = data_host_to_device(test_world, CL_MEM_READ_ONLY, wc2);
    auto cl_bc2 = data_host_to_device(test_world, CL_MEM_READ_ONLY, bc2);
    auto cl_wd1 = data_host_to_device(test_world, CL_MEM_READ_ONLY, wd1);
    auto cl_bd1 = data_host_to_device(test_world, CL_MEM_READ_ONLY, bd1);
    auto cl_wdo = data_host_to_device(test_world, CL_MEM_READ_ONLY, wdo);
    auto cl_bdo = data_host_to_device(test_world, CL_MEM_READ_ONLY, bdo);
    std::cout << "OpenCL Run Model Weights and Biases are Extracted!\n";

    auto conv1_out = emptyClDataBlob<float>(test_world, {24, 24, 32}, CL_MEM_READ_WRITE);
    auto pool1_out = emptyClDataBlob<float>(test_world, {12, 12, 32}, CL_MEM_READ_WRITE);
    auto conv2_out = emptyClDataBlob<float>(test_world, {8, 8, 64}, CL_MEM_READ_WRITE);
    auto pool2_out = emptyClDataBlob<float>(test_world, {4, 4, 64}, CL_MEM_READ_WRITE);
    auto dens1_out = emptyClDataBlob<float>(test_world, {256}, CL_MEM_READ_WRITE);
    auto class_out = emptyClDataBlob<float>(test_world, {10}, CL_MEM_READ_WRITE);
    auto softm_out = emptyClDataBlob<float>(test_world, {10}, CL_MEM_WRITE_ONLY);
    std::cout << "Intermediate data created!" << std::endl;

    cl_uchar conv1_in_width   = static_cast<cl_uchar>(cl_img.dims[0]);
    cl_uchar conv1_in_height  = static_cast<cl_uchar>(cl_img.dims[1]);
    cl_uchar conv1_mask_depth = static_cast<cl_uchar>(cl_img.dims[2]);
    cl_uchar conv2_in_width   = static_cast<cl_uchar>(pool1_out.dims[0]);
    cl_uchar conv2_in_height  = static_cast<cl_uchar>(pool1_out.dims[1]);
    cl_uchar conv2_mask_depth = static_cast<cl_uchar>(pool1_out.dims[2]);

    clSetKernelArg(conv, 0, sizeof(cl_mem), &cl_img.buffer); 
    clSetKernelArg(conv, 1, sizeof(cl_mem), &conv1_out.buffer);
    clSetKernelArg(conv, 2, sizeof(cl_mem), &cl_wc1.buffer);
    clSetKernelArg(conv, 3, sizeof(cl_mem), &cl_bc1.buffer);
    clSetKernelArg(conv, 4, sizeof(cl_uchar), &conv1_in_width);
    clSetKernelArg(conv, 5, sizeof(cl_uchar), &conv1_in_height);
    clSetKernelArg(conv, 6, sizeof(cl_uchar), &conv1_mask_depth);

    StopWatch<> timer;

    size_t global[3] = {24, 24, 32};
    auto t_conv1 = launch_kernel(test_world, conv, global, conv_reqd_wg.data());

    clSetKernelArg(maxp, 0, sizeof(cl_mem), &conv1_out.buffer);
    clSetKernelArg(maxp, 1, sizeof(cl_mem), &pool1_out.buffer);

    global[0] = 12; global[1] = 12; global[2] = 32;
    auto t_pool1 = launch_kernel(test_world, maxp, global, maxp_reqd_wg.data());

    clSetKernelArg(conv, 0, sizeof(cl_mem), &pool1_out.buffer);
    clSetKernelArg(conv, 1, sizeof(cl_mem), &conv2_out.buffer);
    clSetKernelArg(conv, 2, sizeof(cl_mem), &cl_wc2.buffer);
    clSetKernelArg(conv, 3, sizeof(cl_mem), &cl_bc2.buffer);
    clSetKernelArg(conv, 4, sizeof(cl_uchar), &conv2_in_width);
    clSetKernelArg(conv, 5, sizeof(cl_uchar), &conv2_in_height);
    clSetKernelArg(conv, 6, sizeof(cl_uchar), &conv2_mask_depth);

    global[0] = 8; global[1] = 8; global[2] = 64;
    auto t_conv2 = launch_kernel(test_world, conv, global, conv_reqd_wg.data());

    clSetKernelArg(maxp, 0, sizeof(cl_mem), &conv2_out.buffer);
    clSetKernelArg(maxp, 1, sizeof(cl_mem), &pool2_out.buffer);

    global[0] = 4; global[1] = 4; global[2] = 64;
    auto t_pool2 = launch_kernel(test_world, maxp, global, maxp_reqd_wg.data());

    cl_ushort in_neuron1 = static_cast<cl_ushort>(4 * 4 * 64);

    clSetKernelArg(fc, 0, sizeof(cl_mem), &pool2_out.buffer);
    clSetKernelArg(fc, 1, sizeof(cl_mem), &dens1_out.buffer);
    clSetKernelArg(fc, 2, sizeof(cl_mem), &cl_wd1.buffer);
    clSetKernelArg(fc, 3, sizeof(cl_mem), &cl_bd1.buffer);
    clSetKernelArg(fc, 4, sizeof(cl_ushort), &in_neuron1);

    global[0] = 256; global[1] = 1; global[2] = 1;
    auto t_fc1 = launch_kernel(test_world, fc, global, fc_reqd_wg.data());

    cl_ushort in_neuron2 = static_cast<cl_ushort>(256);

    clSetKernelArg(fc, 0, sizeof(cl_mem), &dens1_out.buffer);
    clSetKernelArg(fc, 1, sizeof(cl_mem), &class_out.buffer);
    clSetKernelArg(fc, 2, sizeof(cl_mem), &cl_wdo.buffer);
    clSetKernelArg(fc, 3, sizeof(cl_mem), &cl_bdo.buffer);
    clSetKernelArg(fc, 4, sizeof(cl_ushort), &in_neuron2);

    global[0] = 10; global[1] = 1; global[2] = 1;
    auto t_fc2 = launch_kernel(test_world, fc, global, fc_reqd_wg.data());

    clSetKernelArg(softmax, 0, sizeof(cl_mem), &class_out.buffer);
    clSetKernelArg(softmax, 1, sizeof(cl_mem), &softm_out.buffer);

    global[0] = 1; global[1] = 1; global[2] = 1;
    auto t_soft = launch_kernel(test_world, softmax, global, softmax_reqd_wg.data());

    auto fpga_elapsed = timer.stop();

    auto cnn_outs = data_device_to_host(test_world, softm_out);

    std::cout << "Total Time Elapsed: " << (std::size_t)fpga_elapsed << "\tus"  << '\n';
    std::cout << "FPGA Elapsed Timings: \n";
    std::cout << "conv1: " << (std::size_t)t_conv1 / 1000 << "\tus"  << '\n';
    std::cout << "pool1: " << (std::size_t)t_pool1 / 1000 << "\tus"  << '\n';
    std::cout << "conv2: " << (std::size_t)t_conv2 / 1000 << "\tus"  << '\n';
    std::cout << "pool2: " << (std::size_t)t_pool2 / 1000 << "\tus"  << '\n';
    std::cout << "fc1  : " << (std::size_t)t_fc1   / 1000 << "\tus"  << '\n';
    std::cout << "fc2  : " << (std::size_t)t_fc2   / 1000 << "\tus"  << '\n';
    std::cout << "soft : " << (std::size_t)t_soft  / 1000 << "\tus"  << '\n';
    std::cout << std::endl;

    std::size_t class_no = 0;
    std::cout << std::fixed << std::setprecision(3);
    for(auto c : cnn_outs.buffer)
    {
        std::cout << "Number: " << class_no << "\t\t\t Confidence: %" << c * 100 << '\n';
        ++class_no;
    }

    std::cout << std::scientific << std::setprecision(6) << std::endl;

    clReleaseMemObject(cl_img.buffer);
    clReleaseMemObject(cl_wc1.buffer);
    clReleaseMemObject(cl_bc1.buffer);
    clReleaseMemObject(cl_wc2.buffer);
    clReleaseMemObject(cl_bc2.buffer);
    clReleaseMemObject(cl_wd1.buffer);
    clReleaseMemObject(cl_bd1.buffer);
    clReleaseMemObject(cl_wdo.buffer);
    clReleaseMemObject(cl_bdo.buffer);
    return cnn_outs;
//    return img; // Placeholder
}


float cnn_test::test_img()
{
    Data img = getTestImg();
    //auto img_data = mnist_test::get_a_img("lenet_data/test_img1_7");
    //Data img;
    //img.dims = {28, 28, 1};
    //img.buffer = img_data;

    std::cout << "test image generated!" << std::endl;

    auto seq_out = seq_img_test(img);
    auto ocl_out = ocl_img_test(img);

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
