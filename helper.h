#pragma once

#include "xcl.h"
#include "DataBlob.h"
#include "clDataBlob.h"

#include <string>
#include <vector>
#include <map>

std::string get_kernel_name(cl_kernel k);

std::vector<size_t> get_kernel_reqd_wg_size(cl_kernel k, cl_device_id id);

double launch_kernel(xcl_world world, cl_kernel kernel, size_t global[3], size_t local[3]);

std::vector<cl_kernel> get_kernels_binary(xcl_world world, const char * bin_name,
                                          const std::vector<std::string> & kernel_names);

std::map<std::string, cl_kernel> get_kernel_map(const std::vector<cl_kernel>& kernel_vec);

cl_kernel get_kernel_from_vec(const std::vector<cl_kernel>& kernels, const std::string & name);

template<typename T>
clDataBlob<T> data_host_to_device(xcl_world world, cl_mem_flags flag, DataBlob<T> & blob)
{
    clDataBlob<T> device_data = emptyClDataBlob<T>(world, blob.dims, flag);
    std::size_t tot_size = 1;
    for(auto size : blob.dims)
    {
        tot_size *= size;
    }    
    xcl_memcpy_to_device(world, device_data.buffer, blob.buffer.data(), tot_size * sizeof(T));
    return device_data;
}

template<typename T>
DataBlob<T> data_device_to_host(xcl_world world, clDataBlob<T> & blob)
{
    DataBlob<T> host_data = emptyDataBlob<T>(blob.dims);
    std::size_t tot_size = 1;
    for(auto size : blob.dims)
    {
        tot_size *= size;
    }
    xcl_memcpy_from_device(world, host_data.buffer.data(), blob.buffer, tot_size * sizeof(T));
    return host_data;
}
