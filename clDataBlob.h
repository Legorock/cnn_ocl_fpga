#pragma once

#include <vector>
#include "xcl.h"

template<typename T>
struct clDataBlob
{
    cl_mem buffer;
    cl_mem_flags mem_flag;
    std::vector<std::size_t> dims;
    
    typedef T DataType;
};

template<typename T>
std::size_t getTotalSize(const clDataBlob<T> & dblob)
{
    std::size_t tot_size = 1;
    for(auto size : dblob.dims)
    {
        tot_size *= size;
    }
    return tot_size;
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
