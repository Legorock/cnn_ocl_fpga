#pragma once

#include <vector>

template<typename T>
struct DataBlob
{
    std::vector<T> buffer;
    std::vector<std::size_t> dims;

    typedef T DataType;
};

