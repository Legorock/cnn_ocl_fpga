#include "helper.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>

static void* smalloc(size_t size) 
{
    void* ptr;
    ptr = malloc(size);

    if (ptr == NULL) {
        printf("Error: Cannot allocate memory\n");
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static int load_file_to_memory(const char *filename, char **result) 
{
    unsigned int size;

    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        *result = NULL;
        printf("Error: Could not read file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);

    *result = (char *) smalloc(sizeof(char)*(size+1));

    if (size != fread(*result, sizeof(char), size, f)) {
        free(*result);
        printf("Error: read of kernel failed\n");
        exit(EXIT_FAILURE);
    }

    fclose(f);
    (*result)[size] = 0;

    return size;
}

double launch_kernel(xcl_world world, cl_kernel krnl,
                     size_t global[3], size_t local[3])
{
    cl_event event;
    unsigned long start, stop;

    cl_int err = clEnqueueNDRangeKernel(world.command_queue, krnl, 3,
                                        nullptr, global, local, 0, nullptr, &event);

    if(err != CL_SUCCESS)
    {
        std::cerr << "ERROR: failed executing the kernel!" << '\t' << err << '\n';
        std::cerr << "Error flag: " << oclErrorCode(err) << '\n'; 
        std::cerr << "Kernel Name: " << get_kernel_name(krnl) << '\n';
        exit(EXIT_FAILURE);
    }
    clFinish(world.command_queue);

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(unsigned long),
                            &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(unsigned long),
                            &stop, NULL);
    return (double)stop - (double)start;
}

cl_event launch_kernel_async(xcl_world world, cl_kernel krnl,
                     size_t global[3], size_t local[3])
{
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(world.command_queue, krnl, 3,
                                        nullptr, global, local, 0, nullptr, &event);

    if(err != CL_SUCCESS)
    {
        std::cerr << "ERROR: failed executing the kernel!" << '\t' << err << '\n';
        std::cerr << "Error flag: " << oclErrorCode(err) << '\n'; 
        std::cerr << "Kernel Name: " << get_kernel_name(krnl) << '\n';
        exit(EXIT_FAILURE);
    }
    return event;
}

std::vector<cl_kernel> get_kernels_binary(xcl_world world, const char * bin_name, const std::vector<std::string>& kernel_names)
{
    cl_int err;
    char * krnl_bin;

    const size_t krnl_size = load_file_to_memory(bin_name, &krnl_bin);
    std::cout << "Binary Size: " << krnl_size << std::endl;
    cl_program program = clCreateProgramWithBinary(world.context, 1,
                                                   &world.device_id, &krnl_size,
                                                   (const unsigned char**) &krnl_bin, NULL, &err);

    if ((!program) || (err!=CL_SUCCESS)) {
        printf("Error: Failed to create compute program from binary %d!\n", err);
        printf("Error Flag: %s\n", oclErrorCode(err));
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, world.device_id, CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Error Flag: %s\n", oclErrorCode(err));
        exit(EXIT_FAILURE);
    }

    std::vector<cl_kernel> kernels;
    kernels.reserve(kernel_names.size());

    for(std::size_t i = 0; i < kernel_names.size(); ++i)
    {
        std::string krnl_name = kernel_names[i];
        cl_kernel kernel = clCreateKernel(program, krnl_name.c_str(), &err);
        if (!kernel || err != CL_SUCCESS) {
            printf("Error: Failed to create kernel for %s: %d\n", krnl_name.c_str(), err);
            printf("Error Flag: %s\n", oclErrorCode(err));
            exit(EXIT_FAILURE);
        }
        kernels.push_back(kernel);
    }
    free(krnl_bin);
    return kernels;
}

std::vector<size_t> get_kernel_reqd_wg_size(cl_kernel k, cl_device_id d)
{
    cl_int err;
    size_t wg[3];

    err = clGetKernelWorkGroupInfo(k, d, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(size_t[3]), &wg[0], nullptr);
    if(err != CL_SUCCESS)
    {
        std::cerr << "clGetKernelWorkGroupInfo failed getting compile work group size!\n";
        std::cerr << "Error Code: " << err << '\n';
        std::cerr << "Name of the problematic kernel: " << get_kernel_name(k) << '\n';
        exit(EXIT_FAILURE);
    }
    return std::vector<size_t>{wg[0], wg[1], wg[2]};
}

std::string get_kernel_name(cl_kernel k)
{
    cl_int err;
    size_t name_length;
    std::string kname;

    err = clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &name_length);
    if(err != CL_SUCCESS)
    {
        std::cerr << "clGetKernelInfo failed getting kernel name length!\n";
        exit(EXIT_FAILURE);
    }
    kname.resize(name_length,'\0');
    err = clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, name_length, &kname[0], nullptr);
    if(err != CL_SUCCESS)
    {
        std::cerr << "clGetKernelInfo failed getting kernel name!\n";
        exit(EXIT_FAILURE);
    }
    return kname;
}

std::map<std::string, cl_kernel> get_kernel_map(const std::vector<cl_kernel>& kernel_vec)
{
    std::map<std::string, cl_kernel> kernel_map;
    for(auto & k : kernel_vec)
    {
        auto kernel_name = get_kernel_name(k);
        kernel_map.insert(std::make_pair(kernel_name, k));
        //kernel_map[kernel_name] = k;
    }
    return kernel_map;
}

cl_kernel get_kernel_from_vec(const std::vector<cl_kernel>& kernels, const std::string & name)
{
    for(auto k : kernels)
    {
        auto kname = get_kernel_name(k);
//        if(kname == name)
        if(kname.find(name) != std::string::npos)
        {
            return k;
        }
    }

    std::cerr << "the kernel doesn't exist in the kernels(vector)!\n";
    std::exit(EXIT_FAILURE);
    return nullptr;
}

void print_classes(const std::vector<float> & cls)
{
    std::size_t class_no = 0;
    std::cout << std::fixed << std::setprecision(3);
    for(auto c : cls)
    {
        std::cout << "Number: " << class_no << "\t\t\t Confidence: %" << c * 100 << '\n';
        ++class_no;
    }
    std::cout << std::scientific << std::setprecision(6) << std::endl;
}

