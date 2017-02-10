#pragma once

#include "xcl.h"

#include <string>
#include <vector>
#include <map>


std::string get_kernel_name(cl_kernel k);

std::vector<size_t> get_kernel_reqd_wg_size(cl_kernel k, cl_device_id id);

double launch_kernel(xcl_world world, cl_kernel kernel, size_t global[3], size_t local[3]);
cl_event launch_kernel_async(xcl_world world, cl_kernel kernel, size_t global[3], size_t local[3]);

std::vector<cl_kernel> get_kernels_binary(xcl_world world, const char * bin_name,
                                          const std::vector<std::string> & kernel_names);

std::map<std::string, cl_kernel> get_kernel_map(const std::vector<cl_kernel>& kernel_vec);

cl_kernel get_kernel_from_vec(const std::vector<cl_kernel>& kernels, const std::string & name);

const char *oclErrorCode(cl_int code);

