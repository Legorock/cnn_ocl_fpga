#pragma once

#include <vector>

template<typename T = float>
std::vector<T> gen1Data(const std::size_t length)
{
    std::vector<T> data;
    data.resize(length);
    for(std::size_t i = 0; i < data.size(); ++i)
    {
       data[i] = i;
    }
    return data;
}

template<char dim, typename T = float>
std::vector<T> gen2Data(const std::size_t width, const std::size_t height)
{
    std::vector<T> data;
    data.resize(width * height);
    for(std::size_t h = 0; h < height; ++h)
    {
        for(std::size_t w = 0; w < width; ++w)
        {
            size_t idx = w + h * width;
            switch(dim)
            {
            case 'w': data[idx] = w; break;
            case 'h': data[idx] = h; break;
            default:  data[idx] = (T)dim;
            }
        }
    }
    return data;
}

template<char dim, typename T = float>
std::vector<T> gen3Data(const std::size_t width, const std::size_t height, const std::size_t depth)
{
    std::vector<T> data;
    data.resize(width * height * depth);
    for(std::size_t d = 0; d < depth; ++d)
    {
        for(std::size_t h = 0; h < height; ++h)
        {
            for(std::size_t w = 0; w < width; ++w)
            {
                 size_t idx = w + h * width + d * width * height;
                 switch(dim)
                 {
                 case 'w': data[idx] = w; break;
                 case 'h': data[idx] = h; break;
                 case 'd': data[idx] = d; break;
                 default: data[idx] = (T)dim;
                 }
            }
        }
    }
    return data;
}

template<char dim, typename T = float>
std::vector<T> gen4Data(const std::size_t width, const std::size_t height, const std::size_t depth, const std::size_t slice)
{
    std::vector<T> data;
    data.resize(width * height * depth * slice);
    for(std::size_t s = 0; s < slice; ++s)
    {
        for(std::size_t d = 0; d < depth; ++d)
        {
            for(std::size_t h = 0; h < height; ++h)
            {
                for(std::size_t w = 0; w < width; ++w)
                {
                    std::size_t idx = w + h * width + d * width * height + s * width * height * depth;
                    switch(dim)
                    {
                    case 'w': data[idx] = w; break;
                    case 'h': data[idx] = h; break;
                    case 'd': data[idx] = d; break;
                    case 's': data[idx] = s; break;
                    default: data[idx] = (T)dim;
                    }
                }
            }
        }
    }
    return data;
}
