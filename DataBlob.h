#pragma once

#include <vector>
#include "xcl.h"

template<typename T>
struct DataBlob
{
    std::vector<T> buffer;
    std::vector<std::size_t> dims;

    typedef T DataType;
};

template<typename T>
struct clDataBlob
{
    cl_mem buffer;
    cl_mem_flags mem_flag;
    std::vector<std::size_t> dims;
    
    typedef T DataType;
};

template<typename T>
std::size_t getTotalSize(const T & dblob)
{
    std::size_t tot_size = 1;
    for(auto size : dblob.dims)
    {
        tot_size *= size;
    }
    return tot_size;
}

template<typename T>
DataBlob<T> emptyDataBlob(std::vector<std::size_t> dims, T val = 0)
{
    DataBlob<T> d;
    d.dims = dims;
    std::size_t tot_dim = 1;
    for(auto dd : dims) tot_dim *= dd;
    d.buffer.resize(tot_dim, val);
    return d;
}

template<typename T>
clDataBlob<T> emptyClDataBlob(xcl_world world, const std::vector<std::size_t> dims, cl_mem_flags flag = CL_MEM_READ_WRITE)
{
    clDataBlob<T> dblob;
    dblob.dims = dims;
    dblob.mem_flag = flag;
    auto tot_size = getTotalSize(dblob);

    cl_mem buf_ocl = xcl_malloc(world, flag, sizeof(T) * tot_size);
    dblob.buffer = buf_ocl;
    return dblob;
}

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
