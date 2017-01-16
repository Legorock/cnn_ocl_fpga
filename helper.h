#pragma once

#include "xcl.h"

#include <string>
#include <vector>

std::string get_kernel_name(cl_kernel k);

std::vector<size_t> get_kernel_reqd_wg_size(cl_kernel k, cl_device_id id);

double launch_kernel(xcl_world world, cl_kernel kernel, size_t global[3], size_t local[3]);

std::vector<cl_kernel> get_kernels_binary(xcl_world world, const char * bin_name,
                                          const std::vector<std::string> & kernel_names);


