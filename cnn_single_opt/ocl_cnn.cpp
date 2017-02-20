#include "ocl_cnn.h"

#include <iostream>
#include <iomanip>

#include "helper.h"
#include "Measure.h"


typedef DataBlob<float> Data;
typedef clDataBlob<float> clData;

double getProfileFromEvent(cl_event event)
{
    unsigned long start, stop;

//    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(unsigned long), &start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(unsigned long), &start, nullptr);
//  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(unsigned long), &start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(unsigned long), &stop, nullptr);
    return static_cast<double>(stop) - static_cast<double>(start);
}

//DataBlob<float> ocl_cnn_img_test(DataBlob<float> & img, std::map<std::string, DataBlob<float>> & model, 
//                                 xcl_world & world, const std::vector<cl_kernel> & kernels)
//{
//    //cl_kernel conv = get_kernel_from_vec(kernels, "conv_local");
//    //auto conv_reqd_wg = get_kernel_reqd_wg_size(conv, world.device_id);
//    //cl_kernel maxp = get_kernel_from_vec(kernels, "max_pool2");
//    //auto maxp_reqd_wg = get_kernel_reqd_wg_size(maxp, world.device_id);
//    //cl_kernel fc = get_kernel_from_vec(kernels, "fc");
//    //auto fc_reqd_wg = get_kernel_reqd_wg_size(fc, world.device_id);
//    //cl_kernel softmax = get_kernel_from_vec(kernels, "softmax_layer");
//    //auto softmax_reqd_wg = get_kernel_reqd_wg_size(softmax, world.device_id);
//
//    cl_kernel conv1 = get_kernel_from_vec(kernels, "conv1");
//    auto conv1_wg = get_kernel_reqd_wg_size(conv1, world.device_id);
//    cl_kernel conv2 = get_kernel_from_vec(kernels, "conv2");
//    auto conv2_wg = get_kernel_reqd_wg_size(conv2, world.device_id);
//    cl_kernel mpool1 = get_kernel_from_vec(kernels, "max_pool1");
//    auto mpool1_wg = get_kernel_reqd_wg_size(mpool1, world.device_id);
//    cl_kernel mpool2 = get_kernel_from_vec(kernels, "max_pool2");
//    auto mpool2_wg = get_kernel_reqd_wg_size(mpool2, world.device_id);
//    cl_kernel fc1 = get_kernel_from_vec(kernels, "fc1");
//    auto fc1_wg = get_kernel_reqd_wg_size(fc1, world.device_id);
//    cl_kernel fc2 = get_kernel_from_vec(kernels, "fc2");
//    auto fc2_wg = get_kernel_reqd_wg_size(fc2, world.device_id);
//    cl_kernel softmax = get_kernel_from_vec(kernels, "softmax_layer");
//    auto softmax_reqd_wg = get_kernel_reqd_wg_size(softmax, world.device_id);
//
//    std::cout << "Transfering image to device...\n";
//    auto cl_img = data_host_to_device(world, CL_MEM_READ_ONLY, img);
//    std::cout << "Image is transfered to device\n";
//    auto cl_wc1 = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("wc1"));  // Conv1 weights
//    auto cl_bc1 = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("bc1"));  // Conv1 biases
//    auto cl_wc2 = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("wc2"));  // Conv2 weights
//    auto cl_bc2 = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("bc2"));  // Conv2 biases
//    auto cl_wd1 = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("wd1"));  // FullyConn. (dense) weights
//    auto cl_bd1 = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("bd1"));  // FullyConn. (dense) biases
//    auto cl_wdo = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("wdo"));  // FullyConn. out (dense) weights
//    auto cl_bdo = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("bdo"));  // FullyConn. out (dense) biases
//    std::cout << "OpenCL Run Model Weights and Biases are Extracted!\n";
//
//    auto conv1_out = emptyClDataBlob<float>(world, {24, 24, 32}, CL_MEM_READ_WRITE);
//    auto pool1_out = emptyClDataBlob<float>(world, {12, 12, 32}, CL_MEM_READ_WRITE);
//    auto conv2_out = emptyClDataBlob<float>(world, {8, 8, 64}, CL_MEM_READ_WRITE);
//    auto pool2_out = emptyClDataBlob<float>(world, {4, 4, 64}, CL_MEM_READ_WRITE);
//    auto dens1_out = emptyClDataBlob<float>(world, {256}, CL_MEM_READ_WRITE);
//    auto class_out = emptyClDataBlob<float>(world, {10}, CL_MEM_READ_WRITE);
//    auto softm_out = emptyClDataBlob<float>(world, {10}, CL_MEM_WRITE_ONLY);
//    std::cout << "Intermediate data created!" << std::endl;
//
//    cl_uchar conv1_in_width   = static_cast<cl_uchar>(cl_img.dims[0]);
//    cl_uchar conv1_in_height  = static_cast<cl_uchar>(cl_img.dims[1]);
//    cl_uchar conv1_mask_depth = static_cast<cl_uchar>(cl_img.dims[2]);
//    cl_uchar conv2_in_width   = static_cast<cl_uchar>(pool1_out.dims[0]);
//    cl_uchar conv2_in_height  = static_cast<cl_uchar>(pool1_out.dims[1]);
//    cl_uchar conv2_mask_depth = static_cast<cl_uchar>(pool1_out.dims[2]);
//
//    clSetKernelArg(conv1, 0, sizeof(cl_mem), &cl_img.buffer); 
//    clSetKernelArg(conv1, 1, sizeof(cl_mem), &conv1_out.buffer);
//    clSetKernelArg(conv1, 2, sizeof(cl_mem), &cl_wc1.buffer);
//    clSetKernelArg(conv1, 3, sizeof(cl_mem), &cl_bc1.buffer);
//    clSetKernelArg(conv1, 4, sizeof(cl_uchar), &conv1_in_width);
//    clSetKernelArg(conv1, 5, sizeof(cl_uchar), &conv1_in_height);
//    clSetKernelArg(conv1, 6, sizeof(cl_uchar), &conv1_mask_depth);
//
//    StopWatch<> timer;
//
//    size_t global[3] = {24, 24, 32};
//    auto t_conv1 = launch_kernel_async(world, conv, global, conv_reqd_wg.data());
//
//    clSetKernelArg(maxp, 0, sizeof(cl_mem), &conv1_out.buffer);
//    clSetKernelArg(maxp, 1, sizeof(cl_mem), &pool1_out.buffer);
//
//    global[0] = 12; global[1] = 12; global[2] = 32;
//    auto t_pool1 = launch_kernel_async(world, maxp, global, maxp_reqd_wg.data());
//
//    clSetKernelArg(conv2, 0, sizeof(cl_mem), &pool1_out.buffer);
//    clSetKernelArg(conv2, 1, sizeof(cl_mem), &conv2_out.buffer);
//    clSetKernelArg(conv2, 2, sizeof(cl_mem), &cl_wc2.buffer);
//    clSetKernelArg(conv2, 3, sizeof(cl_mem), &cl_bc2.buffer);
//    clSetKernelArg(conv2, 4, sizeof(cl_uchar), &conv2_in_width);
//    clSetKernelArg(conv2, 5, sizeof(cl_uchar), &conv2_in_height);
//    clSetKernelArg(conv, 6, sizeof(cl_uchar), &conv2_mask_depth);
//
//    global[0] = 8; global[1] = 8; global[2] = 64;
//    auto t_conv2 = launch_kernel_async(world, conv, global, conv_reqd_wg.data());
//
//    clSetKernelArg(maxp, 0, sizeof(cl_mem), &conv2_out.buffer);
//    clSetKernelArg(maxp, 1, sizeof(cl_mem), &pool2_out.buffer);
//
//    global[0] = 4; global[1] = 4; global[2] = 64;
//    auto t_pool2 = launch_kernel_async(world, maxp, global, maxp_reqd_wg.data());
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
//    auto t_fc1 = launch_kernel_async(world, fc, global, fc_reqd_wg.data());
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
//    auto t_fc2 = launch_kernel_async(world, fc, global, fc_reqd_wg.data());
//
//    clSetKernelArg(softmax, 0, sizeof(cl_mem), &class_out.buffer);
//    clSetKernelArg(softmax, 1, sizeof(cl_mem), &softm_out.buffer);
//
//    global[0] = 1; global[1] = 1; global[2] = 1;
//    auto t_soft = launch_kernel_async(world, softmax, global, softmax_reqd_wg.data());
//
//    clFinish(world.command_queue);
//
//    auto fpga_elapsed = timer.stop();
//
//    auto cnn_outs = data_device_to_host(world, softm_out);
//    clFinish(world.command_queue);
//
////    std::cout << "Total Time Elapsed: " << (std::size_t)fpga_elapsed << "\tus"  << '\n';
////    std::cout << "FPGA Elapsed Timings: \n";
////    std::cout << "conv1: " << (std::size_t)getProfileFromEvent(t_conv1) / 1000 << "\tus"  << '\n';
////    std::cout << "pool1: " << (std::size_t)getProfileFromEvent(t_pool1) / 1000 << "\tus"  << '\n';
////    std::cout << "conv2: " << (std::size_t)getProfileFromEvent(t_conv2) / 1000 << "\tus"  << '\n';
////    std::cout << "pool2: " << (std::size_t)getProfileFromEvent(t_pool2) / 1000 << "\tus"  << '\n';
////    std::cout << "fc1  : " << (std::size_t)getProfileFromEvent(t_fc1)   / 1000 << "\tus"  << '\n';
////    std::cout << "fc2  : " << (std::size_t)getProfileFromEvent(t_fc2)   / 1000 << "\tus"  << '\n';
////    std::cout << "soft : " << (std::size_t)getProfileFromEvent(t_soft)  / 1000 << "\tus"  << '\n';
////    std::cout << std::endl;
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

oclCNN::oclCNN(std::map<std::string, DataBlob<float>> & model, xcl_world &  world, const std::vector<cl_kernel> & kernels)
    : m_model(model), m_world(world)
{
    cl_kernel load_model = get_kernel_from_vec(kernels, "load_model_ocm");
    auto loadm_wg = get_kernel_reqd_wg_size(load_model, world.device_id);

    conv1 = get_kernel_from_vec(kernels, "conv1");
    conv1_wg = get_kernel_reqd_wg_size(conv1, world.device_id);
    conv2 = get_kernel_from_vec(kernels, "conv2");
    conv2_wg = get_kernel_reqd_wg_size(conv2, world.device_id);
    mpool1 = get_kernel_from_vec(kernels, "max_pool1");
    mpool1_wg = get_kernel_reqd_wg_size(mpool1, world.device_id);
    mpool2 = get_kernel_from_vec(kernels, "max_pool2");
    mpool2_wg = get_kernel_reqd_wg_size(mpool2, world.device_id);
    fc1 = get_kernel_from_vec(kernels, "fc1");
    fc1_wg = get_kernel_reqd_wg_size(fc1, world.device_id);
    fc2 = get_kernel_from_vec(kernels, "fc2");
    fc2_wg = get_kernel_reqd_wg_size(fc2, world.device_id);
    soft = get_kernel_from_vec(kernels, "softmax_layer");
    soft_wg = get_kernel_reqd_wg_size(soft, world.device_id);
    std::cout << "OpenCL Kernels and Required Workgroup sizes are Extracted!\n";

    cl_wc1 = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("wc1"));  // Conv1 weights
    cl_bc1 = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("bc1"));  // Conv1 biases
    cl_wc2 = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("wc2"));  // Conv2 weights
    cl_bc2 = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("bc2"));  // Conv2 biases
    cl_wd1 = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("wd1"));  // FullyConn. (dense) weights
    cl_bd1 = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("bd1"));  // FullyConn. (dense) biases
    cl_wdo = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("wdo"));  // FullyConn. out (dense) weights
    cl_bdo = data_host_to_device(world, CL_MEM_READ_ONLY, model.at("bdo"));  // FullyConn. out (dense) biases
    std::cout << "CNN Model Parameters are Extracted and Transfered to OpenCL Device!\n";

    clFinish(world.command_queue);

    clSetKernelArg(load_model, 0, sizeof(cl_mem), &cl_wc1.buffer);
    clSetKernelArg(load_model, 1, sizeof(cl_mem), &cl_bc1.buffer);
    clSetKernelArg(load_model, 2, sizeof(cl_mem), &cl_wc2.buffer);
    clSetKernelArg(load_model, 3, sizeof(cl_mem), &cl_bc2.buffer);
    clSetKernelArg(load_model, 4, sizeof(cl_mem), &cl_wd1.buffer);
    clSetKernelArg(load_model, 5, sizeof(cl_mem), &cl_bd1.buffer);
    clSetKernelArg(load_model, 6, sizeof(cl_mem), &cl_wdo.buffer);
    clSetKernelArg(load_model, 7, sizeof(cl_mem), &cl_bdo.buffer);

    size_t global[3] = {1, 1, 1};
    auto t_load_model = launch_kernel(world, load_model, global, loadm_wg.data());
    std::cout << "CNN Model Parameters are Relocated to On-Chip Constant Memory!\n";

    conv1_out = emptyClDataBlob<float>(world, {24, 24, 32}, CL_MEM_READ_WRITE);
    pool1_out = emptyClDataBlob<float>(world, {12, 12, 32}, CL_MEM_READ_WRITE);
    conv2_out = emptyClDataBlob<float>(world, {8, 8, 64}, CL_MEM_READ_WRITE);
    pool2_out = emptyClDataBlob<float>(world, {4, 4, 64}, CL_MEM_READ_WRITE);
    dens1_out = emptyClDataBlob<float>(world, {256}, CL_MEM_READ_WRITE);
    class_out = emptyClDataBlob<float>(world, {10}, CL_MEM_READ_WRITE);
    softm_out = emptyClDataBlob<float>(world, {10}, CL_MEM_WRITE_ONLY);
    std::cout << "Intermediate data created!\n";


//    clSetKernelArg(conv1, 0, sizeof(cl_mem), &cl_img.buffer); 
    clSetKernelArg(soft, 0, sizeof(cl_mem), &softm_out.buffer);
}

oclCNN::~oclCNN()
{
    clReleaseMemObject(cl_wc1.buffer);
    clReleaseMemObject(cl_bc1.buffer);
    clReleaseMemObject(cl_wc2.buffer);
    clReleaseMemObject(cl_bc2.buffer);
    clReleaseMemObject(cl_wd1.buffer);
    clReleaseMemObject(cl_bd1.buffer);
    clReleaseMemObject(cl_wdo.buffer);
    clReleaseMemObject(cl_bdo.buffer);
    clReleaseMemObject(conv1_out.buffer);
    clReleaseMemObject(pool1_out.buffer);
    clReleaseMemObject(conv2_out.buffer);
    clReleaseMemObject(pool2_out.buffer);
    clReleaseMemObject(dens1_out.buffer);
    clReleaseMemObject(class_out.buffer);
    clReleaseMemObject(softm_out.buffer);
}

DataBlob<float> oclCNN::runImg(DataBlob<float> & img)
{
//    return {{0,0,0,0,0,0,0,1.0,0,0}, {10}};

    size_t global[3];
    auto cl_img = data_host_to_device(m_world, CL_MEM_READ_ONLY, img);
    clSetKernelArg(conv1, 0, sizeof(cl_mem), &cl_img.buffer); 
    StopWatch<> timer;

    global[0] = 24; global[1] = 24; global[2] = 32;
    auto t_conv1 = launch_kernel_async(m_world, conv1, global, conv1_wg.data());
    
    global[0] = 12; global[1] = 12; global[2] = 32;
    auto t_pool1 = launch_kernel_async(m_world, mpool1, global, mpool1_wg.data());

    global[0] = 8; global[1] = 8; global[2] = 64;
    auto t_conv2 = launch_kernel_async(m_world, conv2, global, conv2_wg.data());

    global[0] = 4; global[1] = 4; global[2] = 64;
    auto t_pool2 = launch_kernel_async(m_world, mpool2, global, mpool2_wg.data());

    global[0] = 256; global[1] = 1; global[2] = 1;
    auto t_fc1 = launch_kernel_async(m_world, fc1, global, fc1_wg.data());

    global[0] = 10; global[1] = 1; global[2] = 1;
    auto t_fc2 = launch_kernel_async(m_world, fc2, global, fc2_wg.data());
    
    global[0] = 1; global[1] = 1; global[2] = 1;
    auto t_soft = launch_kernel_async(m_world, soft, global, soft_wg.data());

    clFinish(m_world.command_queue);
    auto fpga_elapsed = timer.stop();

    auto cnn_outs = data_device_to_host(m_world, softm_out);
    clFinish(m_world.command_queue);

    std::cout << std::fixed << "Total Time Elapsed: " << fpga_elapsed << "\tus"  << '\n';
//    std::cout << "FPGA Elapsed Timings: \n";
//    std::cout << "conv1: " << getProfileFromEvent(t_conv1) / 1000 << "\tus"  << '\n';
//    std::cout << "pool1: " << getProfileFromEvent(t_pool1) / 1000 << "\tus"  << '\n';
//    std::cout << "conv2: " << getProfileFromEvent(t_conv2) / 1000 << "\tus"  << '\n';
//    std::cout << "pool2: " << getProfileFromEvent(t_pool2) / 1000 << "\tus"  << '\n';
//    std::cout << "fc1  : " << getProfileFromEvent(t_fc1)   / 1000 << "\tus"  << '\n';
//    std::cout << "fc2  : " << getProfileFromEvent(t_fc2)   / 1000 << "\tus"  << '\n';
//    std::cout << "soft : " << getProfileFromEvent(t_soft)  / 1000 << "\tus"  << '\n';
    std::cout << std::endl;
    return cnn_outs;
}
