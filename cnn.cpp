
#include <iostream>
#include <string>
#include <cstring>

// OpenCL includes
#include "xcl.h"

#include "cnn_test.h"

int main(int argc, char ** argv)
{
// TARGET DEVICE macro needs to be passed from gcc command line
#if defined(SDA_PLATFORM) && !defined(TARGET_DEVICE)
  #define STR_VALUE(arg)      #arg
  #define GET_STRING(name) STR_VALUE(name)
  #define TARGET_DEVICE GET_STRING(SDA_PLATFORM)
#endif
    const char * target_device_name = TARGET_DEVICE;
    const char * target_vendor = "Xilinx";

    if(argc != 3 && argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " <xclbin>" << " <{'test', 'test_img', 'runall <images>', 'run <image>'}>" << std::endl;
	return -1;
    }

    std::string xclbinFilename(argv[1]);
    std::string exe_mode(argv[2]);
    std::string mode_param;

    if(argc == 4)
    {
        mode_param = std::string(argv[3]);
    }

    std::cout << "Vendor: " << target_vendor << '\n'
              << "Device: " << target_device_name << '\n'
              << "XCLBIN: " << xclbinFilename << '\n'
              << "EXEMOD: " << exe_mode << std::endl;

    xcl_world world;
    bool is_binary;
    if(xclbinFilename.find(".xclbin") != std::string::npos)
    {
         std::cout << "OpenCL Accelerator binary initialization!" << std::endl;
         world = xcl_world_single(CL_DEVICE_TYPE_ACCELERATOR, target_vendor, target_device_name);
         std::cout << "OpenCL Accelerator initialization end!" << std::endl;
         is_binary = true;
    }
    else if(xclbinFilename.find(".cl") != std::string::npos)
    {
         std::cout << "OpenCL CPU run initialization!" << std::endl;
         world = xcl_world_single(CL_DEVICE_TYPE_CPU, nullptr, nullptr);
         std::cout << "OpenCL CPU initialization end!" << std::endl;
         is_binary = false;
    }
    else
    {
         std::cerr << "opencl file has unexpected name!\n"
                   << "It should end with .xclbin for xilinx binary or .cl for source compilation\n" 
                   << std::endl;
         return -1;
    }
    
    if(exe_mode == "test") 
    {
        cnn_test t(world, xclbinFilename.c_str(), is_binary);
//        auto test_err = t.test();
//        std::cout << "Test error: " << test_err << std::endl;
    }
    else if(exe_mode == "test_img")
    {
        cnn_test t(world, xclbinFilename.c_str(), is_binary);
        auto test_err = t.test_img();
        std::cout << "Test error: " << test_err << std::endl;
    }
    else if(exe_mode == "runall")
    {

    }
    else if(exe_mode == "run")
    {
    
    }
    else
    {
        std::cerr << "Unknown execution mod!\n";
        return -1;   
    }

    xcl_release_world(world);
    return 0;
}
