#include "cnn_test.h"

#include <cstdlib>
#include <iostream>
#include <fstream>

#include "genData.h"
#include "seq.h"

#include "oclHelper.h"
#include "helper.h"

template<typename T>
inline static void print_buf(std::ostream& o, const T *  buf, 
                             const std::vector<std::size_t>& dims, const std::size_t curr_dim);

const std::vector<std::string> kernel_names = {"max_pool2", "conv_local_flatmem", "softmax_layer", "fully_connected_local"};
const std::vector<std::string> kernel_layers = {"max_pool", "conv", "fully_connected", "softmax"};

inline static krnl_data_map sequential_runs(std::map<std::string, std::vector<Data>>& data)
{
    krnl_data_map outs;
    
    for(auto & krnl_run : data)
    {
        const std::string & krnl_name = krnl_run.first;
        std::vector<Data> & krnl_data = krnl_run.second;
        if(krnl_name == "max_pool")
        {
            //std::cerr << "maxpool sequential test!\n";
            auto v_in_dims = krnl_data[0].dims;
            auto v_out_dims = krnl_data[1].dims;
            std::size_t in_dims[3] = {v_in_dims[0], v_in_dims[1], v_in_dims[2]};
            std::size_t out_dims[3] = {v_out_dims[0], v_out_dims[1], v_out_dims[2]};
            max_pool2_seq(krnl_data[0].buffer, in_dims, krnl_data[1].buffer, out_dims);
            //std::cerr << "maxpool sequential ends!\n";
        }
        else if(krnl_name == "conv")
        {
            //std::cerr << "conv sequential test!\n";
            auto v_in_dims = krnl_data[0].dims;
            auto v_out_dims = krnl_data[1].dims;
            std::size_t in_dims[3] = {v_in_dims[0], v_in_dims[1], v_in_dims[2]};
            std::size_t out_dims[3] = {v_out_dims[0], v_out_dims[1], v_out_dims[2]};
            conv_seq(krnl_data[0].buffer, in_dims, krnl_data[1].buffer, out_dims, krnl_data[2].buffer, krnl_data[3].buffer);
            //std::cerr << "conv sequnetial ends!\n";
        }
        else if(krnl_name == "softmax")
        {
            //std::cerr << "softmax sequential test!\n";
            std::size_t in_dim = krnl_data[0].dims[0];
            softmax_seq(krnl_data[0].buffer, in_dim, krnl_data[1].buffer);
            //std::cerr << "softmax sequential ends!\n";
        }
        else if(krnl_name == "fully_connected")
        {
            //std::cerr << "fc sequential test!\n";
            std::size_t in_dim = krnl_data[0].dims[0];
            std::size_t out_dim = krnl_data[1].dims[0];
            fc_seq(krnl_data[0].buffer, in_dim, krnl_data[1].buffer, out_dim, krnl_data[2].buffer, krnl_data[3].buffer);
            //std::cerr << "fc sequential ends!\n";
        }

        std::cout << krnl_name << " sequential tested!" << std::endl;
        outs[krnl_name] = krnl_data[1];
    }
    return outs;
}

inline static std::map<std::string, std::vector<Data>> gen_run_data()
{
    std::map<std::string, std::vector<Data>> data;
    for(auto & krnl_name : kernel_layers)
    {
        if(krnl_name == "max_pool")
        {
            Data in, out;
            in.dims = {24, 24, 4};
            out.dims = {in.dims[0]/2, in.dims[1]/2, in.dims[2]};
            in.buffer = gen3Data<'d'>(in.dims[0], in.dims[1], in.dims[2]);
            out.buffer.resize(out.dims[0] * out.dims[1] * out.dims[2], 0.0f);

            data[krnl_name] = std::vector<Data>({in, out});
        }
        else if(krnl_name == "conv")
        {
            Data in, out, w, b;
            in.dims = {12, 12, 1};
            out.dims = {in.dims[0]-4, in.dims[1]-4, 32};
            w.dims = {5 * 5 * in.dims[2] * out.dims[2]};
            b.dims = {out.dims[2]};
            in.buffer = gen3Data<'w'>(in.dims[0], in.dims[1], in.dims[2]);
            out.buffer.resize(out.dims[0] * out.dims[1] * out.dims[2], 0.0f);
            w.buffer = gen1Data(w.dims[0]);
            b.buffer = gen1Data(b.dims[0]);

            data[krnl_name] = std::vector<Data>({in, out, w, b});
        }
        else if(krnl_name == "softmax")
        {
            Data in, out;
            in.dims = {10};
            out.dims = {in.dims};
            in.buffer = gen1Data(in.dims[0]);
            out.buffer.resize(out.dims[0], 0.0f);
            
            data[krnl_name] = std::vector<Data>({in, out});
        }
        else if(krnl_name == "fully_connected")
        {
            Data in, out, w, b;
            in.dims = {512};
            out.dims = {256};
            w.dims = {in.dims[0] * out.dims[0]};
            b.dims = {out.dims[0]};
            in.buffer = gen1Data(in.dims[0]);
            out.buffer.resize(out.dims[0], 0.0f);
            w.buffer = std::vector<float>(w.dims[0], 1.0f);
            b.buffer = gen1Data(b.dims[0]);
            
            data[krnl_name] = std::vector<Data>({in, out, w, b});
        }
    }
    return data;
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

void cnn_test::ocl_maxpool2_run(cl_kernel kernel, std::vector<Data>& data)
{
    auto & data_in = data[0];
    auto & data_out = data[1];
    size_t buf_in_size = data_in.dims[0] * data_in.dims[1] * data_in.dims[2];
    size_t buf_out_size = data_out.dims[0] * data_out.dims[1] * data_out.dims[2];

    buf_in_size *= sizeof(float);
    buf_out_size *= sizeof(float);

    // Device buffer allocation
    cl_mem buf_in = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_in_size);
    cl_mem buf_out = xcl_malloc(test_world, CL_MEM_WRITE_ONLY, buf_out_size);

    // HOST to DEVICE transfer
    xcl_memcpy_to_device(test_world, buf_in, data_in.buffer.data(), buf_in_size);

    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
    if(err != CL_SUCCESS)
    {
        std::cerr << "One of the clSetKernelArg failed!\n";
        std::exit(EXIT_FAILURE);
    }

    auto reqd_wg_size = get_kernel_reqd_wg_size(kernel, test_world.device_id);

    size_t global[3] = {data_out.dims[0], data_out.dims[1], data_out.dims[2]};
    size_t local[3] = {reqd_wg_size[0], reqd_wg_size[1], reqd_wg_size[2]};
    std::cout << "global: " << global[0] << '\t' << global[1] << '\t' << global[2] << '\n';
    std::cout << "local: " << local[0] << '\t' << local[1] << '\t' << local[2] << std::endl;

    std::cout << "OpenCL kernel launching!" << std::endl;
    auto exe_t = launch_kernel(test_world, kernel, global, local); 
    std::cout << "OpenCL kernel finished!" << std::endl;

    std::cout << "Max Pooling kernel exe time(ns): " << exe_t << " ns" << std::endl; 

    // DEVICE to HOST transfer
    xcl_memcpy_from_device(test_world, data_out.buffer.data(), buf_out, buf_out_size);

    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
}

void cnn_test::ocl_conv_run(cl_kernel kernel, std::vector<Data>& data)
{
    auto & data_in = data[0];
    auto & data_out = data[1];
    auto & data_weight = data[2];
    auto & data_biases = data[3];
    auto & in_dims = data[0].dims;
    auto & out_dims = data[1].dims;
    size_t buf_in_size = in_dims[0] * in_dims[1] * in_dims[2]  * sizeof(float);
    size_t buf_out_size = out_dims[0] * out_dims[1] * out_dims[2] * sizeof(float);
    size_t buf_w_size = data[2].dims[0] * sizeof(float);
    size_t buf_b_size = data[3].dims[0] * sizeof(float);

    // Device buffer allocation
    cl_mem buf_in = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_in_size);
    cl_mem buf_out = xcl_malloc(test_world, CL_MEM_WRITE_ONLY, buf_out_size);
    cl_mem buf_w = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_w_size);
    cl_mem buf_b = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_b_size);

    // HOST to DEVICE transfer
    xcl_memcpy_to_device(test_world, buf_in, data_in.buffer.data(), buf_in_size);
    xcl_memcpy_to_device(test_world, buf_w, data_weight.buffer.data(), buf_w_size);
    xcl_memcpy_to_device(test_world, buf_b, data_biases.buffer.data(), buf_b_size);

    cl_uchar in_width = static_cast<cl_uchar>(in_dims[0]);
    cl_uchar in_height = static_cast<cl_uchar>(in_dims[1]);
    cl_uchar mask_depth = static_cast<cl_uchar>(in_dims[2]);


    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_w);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_b);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_uchar), &in_width);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_uchar), &in_height);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_uchar), &mask_depth);

    if(err != CL_SUCCESS)
    {
        std::cerr << "One of the clSetKernelArg failed!\n";
        std::exit(EXIT_FAILURE);
    }

    auto reqd_wg_size = get_kernel_reqd_wg_size(kernel, test_world.device_id);

    size_t global[3] = {out_dims[0], out_dims[1], out_dims[2]};
    size_t local[3] = {reqd_wg_size[0], reqd_wg_size[1], reqd_wg_size[2]};
    std::cout << "global: " << global[0] << '\t' << global[1] << '\t' << global[2] << '\n';
    std::cout << "local: " << local[0] << '\t' << local[1] << '\t' << local[2] << std::endl;

    std::cout << "OpenCL kernel launching!" << std::endl;
    auto exe_t = launch_kernel(test_world, kernel, global, local); 
    std::cout << "OpenCL kernel finished!" << std::endl;

    std::cout << "Convolution kernel exe time(ns): " << exe_t << " ns" << std::endl; 

    // DEVICE to HOST transfer
    xcl_memcpy_from_device(test_world, data_out.buffer.data(), buf_out, buf_out_size);

    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
    clReleaseMemObject(buf_w);
    clReleaseMemObject(buf_b);
}

void cnn_test::ocl_fc_run(cl_kernel kernel, std::vector<Data>& data)
{
    auto & data_in = data[0];
    auto & data_out = data[1];
    auto & data_weight = data[2];
    auto & data_biases = data[3];
    size_t buf_in_size = data[0].dims[0] * sizeof(float);
    size_t buf_out_size = data[1].dims[0] * sizeof(float);
    size_t buf_w_size = data[2].dims[0] * sizeof(float);
    size_t buf_b_size = data[3].dims[0] * sizeof(float);

    // Device buffer allocation
    cl_mem buf_in = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_in_size);
    cl_mem buf_out = xcl_malloc(test_world, CL_MEM_WRITE_ONLY, buf_out_size);
    cl_mem buf_w = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_w_size);
    cl_mem buf_b = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_b_size);

    // HOST to DEVICE transfer
    xcl_memcpy_to_device(test_world, buf_in, data_in.buffer.data(), buf_in_size);
    xcl_memcpy_to_device(test_world, buf_w, data_weight.buffer.data(), buf_w_size);
    xcl_memcpy_to_device(test_world, buf_b, data_biases.buffer.data(), buf_b_size);

    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_w);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_b);
    if(err != CL_SUCCESS)
    {
        std::cerr << "One of the clSetKernelArg failed!\n";
        std::exit(EXIT_FAILURE);
    }

    auto reqd_wg_size = get_kernel_reqd_wg_size(kernel, test_world.device_id);

    size_t global[3] = {data[0].dims[0], 1, 1};
    size_t local[3] = {reqd_wg_size[0], reqd_wg_size[1], reqd_wg_size[2]};
    std::cout << "global: " << global[0] << '\t' << global[1] << '\t' << global[2] << '\n';
    std::cout << "local: " << local[0] << '\t' << local[1] << '\t' << local[2] << std::endl;

    std::cout << "OpenCL kernel launching!" << std::endl;
    auto exe_t = launch_kernel(test_world, kernel, global, local); 
    std::cout << "OpenCL kernel finished!" << std::endl;

    std::cout << "FullyConnected kernel exe time(ns): " << exe_t << " ns" << std::endl; 

    // DEVICE to HOST transfer
    xcl_memcpy_from_device(test_world, data_out.buffer.data(), buf_out, buf_out_size);

    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
    clReleaseMemObject(buf_w);
    clReleaseMemObject(buf_b);
}

void cnn_test::ocl_softmax_run(cl_kernel kernel, std::vector<Data>& data)
{
    auto & data_in = data[0];
    auto & data_out = data[1];
    size_t buf_in_size = data[0].dims[0] * sizeof(float);
    size_t buf_out_size = data[1].dims[0] * sizeof(float);

    // Device buffer allocation
    cl_mem buf_in = xcl_malloc(test_world, CL_MEM_READ_ONLY, buf_in_size);
    cl_mem buf_out = xcl_malloc(test_world, CL_MEM_WRITE_ONLY, buf_out_size);

    // HOST to DEVICE transfer
    xcl_memcpy_to_device(test_world, buf_in, data_in.buffer.data(), buf_in_size);

    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
    if(err != CL_SUCCESS)
    {
        std::cerr << "One of the clSetKernelArg failed!\n";
        std::exit(EXIT_FAILURE);
    }

    size_t global[3] = {1,1,1};
    size_t local[3] = {1,1,1};
    std::cout << "global: " << global[0] << '\t' << global[1] << '\t' << global[2] << '\n';
    std::cout << "local: " << local[0] << '\t' << local[1] << '\t' << local[2] << std::endl;

    std::cout << "OpenCL kernel launching!" << std::endl;
    auto exe_t = launch_kernel(test_world, kernel, global, local); 
    std::cout << "OpenCL kernel finished!" << std::endl;

    std::cout << "Softmax kernel exe time(ns): " << exe_t << " ns" << std::endl; 

    // DEVICE to HOST transfer
    xcl_memcpy_from_device(test_world, data_out.buffer.data(), buf_out, buf_out_size);

    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
}

krnl_data_map cnn_test::ocl_runs(std::map<std::string, std::vector<Data>>& data)
{
    krnl_data_map out;

    for(cl_kernel kernel : kernels)
    {
        auto kernel_name = get_kernel_name(kernel);
        std::cout << "actual ocl kernel name: " << kernel_name << std::endl;
        if(kernel_name.find("max_pool") != std::string::npos)
        {
            ocl_maxpool2_run(kernel, data["max_pool"]);
            out["max_pool"] = data["max_pool"][1];
        }
        else if(kernel_name.find("conv") != std::string::npos)
        {
            ocl_conv_run(kernel, data["conv"]);
            out["conv"] = data["conv"][1];
        }
        else if(kernel_name.find("fully_connected") != std::string::npos)
        {
           ocl_fc_run(kernel, data["fully_connected"]);
           out["fully_connected"] = data["fully_connected"][1];
        }
        else if(kernel_name.find("softmax") != std::string::npos)
        {
           ocl_softmax_run(kernel, data["softmax"]);
           out["softmax"] = data["softmax"][1];
        }
        else
        {
            std::cerr << "kernel_name does not match any possibility of layer!\n";
            continue;
        }
    }
    return out;
}

// Constructor
cnn_test::cnn_test(xcl_world& world, const char * clFilename, bool isBinary)
  : test_world(world)
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

void cnn_test::test()
{
    auto seq_data = gen_run_data();
    std::cout << "data generation for test ends!" << std::endl;
    // Sequential CPU run to form a valid output values
    std::cout << "Sequential Runs start!" << std::endl;
    auto seq_out = sequential_runs(seq_data);
    std::cout << "Sequential Runs end!" << std::endl;
    std::ofstream seq_test_out("seq_out.txt");
    print_test_results(seq_test_out, seq_out, "all");
    seq_test_out.close();

    auto ocl_data = gen_run_data();
    std::cout << "data generation for test ends!" << std::endl;
    // OpenCL test run
    std::cout << "OpenCL Runs start!" << std::endl;
    auto ocl_out = ocl_runs(ocl_data);
    std::cout << "OpenCL Runs end!" << std::endl;
    std::ofstream ocl_test_out("ocl_out.txt");
    print_test_results(ocl_test_out, ocl_out, "all");
    ocl_test_out.close();
    
//    std::vector<std::size_t> dim = {5, 3, 2};
//    auto print_test = gen3Data<'0'>(dim[0], dim[1], dim[2]);
//    print_buf<float> (std::cerr, print_test.data(), dim, dim.size()-1);


}
