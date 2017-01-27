#pragma once

#include <vector>

template<typename T>
struct DataBlob
{
    std::vector<T> buffer;
    std::vector<std::size_t> dims;

    typedef T DataType;
};

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
