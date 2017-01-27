#include "seq.h"

#include <iostream>
#include <cmath>

typedef std::size_t num;

inline static num index4(const num x, const num y, const num z, const num w, const num d[4])
{
    return (x + y * d[0] + z * d[0] * d[1] + w * d[0] * d[1] * d[2]);
}

inline static num index3(const num x, const num y, const num z, const num d[3])
{
    return (x + y * d[0] + z * d[0] * d[1]);
}

inline static num index2(const num x, const num y, const num width)
{
    return (x + y * width);
}

inline static float relu(float a)
{
    return (a > 0) ? a : 0;
}

inline static float max(float x, float y)
{
    return (x > y) ? x : y;
}

void max_pool2_seq(const std::vector<float>& in,  const std::size_t in_dims[3],
                         std::vector<float>& out, const std::size_t out_dims[3])
{
    for(num d = 0; d < in_dims[2]; ++d)
    {
        for(num h = 0; h < in_dims[1]; h += 2)
        {
            for(num w = 0; w < in_dims[0]; w += 2)
            {
                num i1 = index3(w, h, d, in_dims);
                num i2 = index3(w+1, h, d, in_dims);
                num i3 = index3(w, h+1, d, in_dims);
                num i4 = index3(w+1, h+1, d, in_dims);
                num o = index3(w/2, h/2, d, out_dims);
                //out[o] = max(i4, max(i3, max(i2, i1)));
                out[o] = max(in[i4], max(in[i3], max(in[i2], in[i1])));
            }
        }
    }
}

void conv_seq(const std::vector<float>& in,      const std::size_t in_dims[3],
                    std::vector<float>& out,     const std::size_t out_dims[3],
              const std::vector<float>& weights, const std::vector<float>& biases)
{
    num wdims[4] = {5, 5, in_dims[2], out_dims[2]};
    for(num ofeat = 0; ofeat < out_dims[2]; ++ofeat)
    {
        for(num oh = 0; oh < out_dims[1]; ++oh)
        {
            for(num ow = 0; ow < out_dims[0]; ++ow)
            {
                float conv = 0.0f;
                for(num ifeat = 0; ifeat < in_dims[2]; ++ifeat)
                {
                    num idx = index3(ow, oh, ifeat, in_dims);
                    num widx = index4(0, 0, ifeat, ofeat, wdims);
                    for(num kh = 0; kh < 5 ; ++kh)
                    {
                        for(num kw = 0; kw < 5; ++kw)
                        {  
                            conv += in[idx + index2(kw, kh, in_dims[0])] * 
                                        weights[widx + index2(kw, kh, 5)];
                        }
                    }
                }
                out[index3(ow, oh, ofeat, out_dims)] = relu(conv + biases[ofeat]);
            }
        }
    }
}

void fc_seq(const std::vector<float>& in,  const std::size_t in_dim,
                  std::vector<float>& out, const std::size_t out_dim,
            const std::vector<float>& w,   const std::vector<float>& b)
{
     for(num o = 0; o < out_dim; ++o)
     {
         float t = 0.0f;
         for(num i = 0; i < in_dim; ++i)
         {
             t += in[i] * w[i + o * in_dim];
         }
         out[o] = relu(t + b[o]);
     }
}

void softmax_seq(const std::vector<float>& in, const std::size_t dim,
                       std::vector<float>& out)
{
    float sum = 0.0f;
    for(num i = 0; i < dim; ++i)
    {
        sum += std::exp(in[i]);
    }

    for(num i = 0; i < dim; ++i)
    {
        out[i] = std::exp(in[i]) / sum;
    }   
}
