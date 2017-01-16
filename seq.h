#pragma once

#include <vector>

void max_pool2_seq(const std::vector<float>& in,  const std::size_t in_dims[3], 
                         std::vector<float>& out, const std::size_t out_dims[3]);

void conv_seq(const std::vector<float>& in,      const std::size_t in_dims[3], 
                    std::vector<float>& out,     const std::size_t out_dims[3],
              const std::vector<float>& weights, const std::vector<float>& biases);

void fc_seq(const std::vector<float>& in,   const std::size_t in_dim,
                  std::vector<float>& out, const std::size_t out_dim,
            const std::vector<float>& w,   const std::vector<float>& b);

void softmax_seq(const std::vector<float>& in, const std::size_t dim,
                       std::vector<float>& out);



