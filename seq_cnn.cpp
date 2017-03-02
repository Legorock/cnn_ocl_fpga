#include "seq_cnn.h"

#include <iostream>
#include <iomanip>

#include "helper.h"
#include "Measure.h"


typedef DataBlob<float> Data;

Data seq_cnn_img_test(const Data& img, const std::map<std::string, DataBlob<float>>& model)
{
    std::cout << "Sequential Run Model Data Loaded!\n";
    auto wc1 = model.at("wc1"); // Conv1 weights
    auto bc1 = model.at("bc1"); // Conv1 biases
    auto wc2 = model.at("wc2"); // Conv2 weights
    auto bc2 = model.at("bc2"); // Conv2 biases
    auto wd1 = model.at("wd1"); // FullyConn. (dense) weights
    auto bd1 = model.at("bd1"); // FullyConn. (dense) biases
    auto wdo = model.at("wdo"); // FullyConn. out (dense) weights
    auto bdo = model.at("bdo"); // FullyConn. out (dense) biases
    std::cout << "Sequential Run Model Weights and Biases are Extracted!\n";
    std::cout << "Creating intermediate data...\n";
    Data conv1_out = emptyDataBlob<float>({24, 24, 32});
    Data pool1_out = emptyDataBlob<float>({12, 12, 32});
    Data conv2_out = emptyDataBlob<float>({8, 8, 64});
    Data pool2_out = emptyDataBlob<float>({4, 4, 64});
    Data dens1_out = emptyDataBlob<float>({256});
    Data class_out = emptyDataBlob<float>({10}); 
    std::cout << "Intermediate data created!" << std::endl; 

    StopWatch<> timer;

    conv_seq(img.buffer, img.dims.data(),
             conv1_out.buffer, conv1_out.dims.data(),
             wc1.buffer, bc1.buffer);

    max_pool2_seq(conv1_out.buffer, conv1_out.dims.data(),
              pool1_out.buffer, pool1_out.dims.data());

    conv_seq(pool1_out.buffer, pool1_out.dims.data(),
             conv2_out.buffer, conv2_out.dims.data(),
             wc2.buffer, bc2.buffer);

    max_pool2_seq(conv2_out.buffer, conv2_out.dims.data(),
                  pool2_out.buffer, pool2_out.dims.data());

    fc_seq(pool2_out.buffer, (pool2_out.dims[0]*pool2_out.dims[1]*pool2_out.dims[2]),
           dens1_out.buffer, dens1_out.dims[0],
           wd1.buffer, bd1.buffer);

    fc_seq(dens1_out.buffer, dens1_out.dims[0],
           class_out.buffer, class_out.dims[0],
           wdo.buffer, bdo.buffer);

    softmax_seq(class_out.buffer, class_out.dims[0], class_out.buffer);

    auto cpu_elapsed = timer.stop();
    std::cout << std::fixed << "Total Elapsed Time: " << cpu_elapsed / 1000.0 << " ms" << '\n';
    return class_out;
}
