#include "seq_cnn.h"

#include <iostream>
#include <iomanip>

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
    StopWatch<> conv1_t;

    conv_seq(img.buffer, img.dims.data(),
             conv1_out.buffer, conv1_out.dims.data(),
             wc1.buffer, bc1.buffer);

    auto s = 0.0f;
    for(auto out : conv1_out.buffer)
    {
        s += out;
    }
    std::cerr << "conv1 out sum: " << s << '\n';

    auto t_conv1 = conv1_t.stop();
    StopWatch<> pool1_t;

    max_pool2_seq(conv1_out.buffer, conv1_out.dims.data(),
              pool1_out.buffer, pool1_out.dims.data());

    s = 0.0f;
    for(auto out : pool1_out.buffer)
    {
        s += out;   
    }
    std::cerr << "pool1 out sum: " << s << '\n';

    auto t_pool1 = pool1_t.stop();
    StopWatch<> conv2_t;

    conv_seq(pool1_out.buffer, pool1_out.dims.data(),
             conv2_out.buffer, conv2_out.dims.data(),
             wc2.buffer, bc2.buffer);

    s = 0.0f;
    for(auto out : conv2_out.buffer)
    {
        s += out;
    }
    std::cerr << "conv2 out sum: " << s << '\n';

    auto t_conv2 = conv2_t.stop();
    StopWatch<> pool2_t;

    max_pool2_seq(conv2_out.buffer, conv2_out.dims.data(),
                  pool2_out.buffer, pool2_out.dims.data());

    s = 0.0f;
    for(auto out : pool2_out.buffer)
    {
        s += out;   
    }
    std::cerr << "pool2 out sum: " << s << '\n';

    auto t_pool2 = pool2_t.stop();
    StopWatch<> fc1_t;

    fc_seq(pool2_out.buffer, (pool2_out.dims[0]*pool2_out.dims[1]*pool2_out.dims[2]),
           dens1_out.buffer, dens1_out.dims[0],
           wd1.buffer, bd1.buffer);
    
    s = 0.0f;
    for(auto out : dens1_out.buffer)
    {
        s += out;
    }
    std::cerr << "dens1 out sum: " << s << '\n';

    auto t_fc1 = fc1_t.stop();
    StopWatch<> fc2_t;

    fc_seq(dens1_out.buffer, dens1_out.dims[0],
           class_out.buffer, class_out.dims[0],
           wdo.buffer, bdo.buffer);

    auto t_fc2 = fc2_t.stop();
    StopWatch<> softmax_t;

    softmax_seq(class_out.buffer, class_out.dims[0], class_out.buffer);

    auto t_soft = softmax_t.stop();
    auto cpu_elapsed = timer.stop();
    std::cout << "Total Elapsed Time: " << cpu_elapsed << " us" << '\n';
    std::cout << "CPU Elapsed Timings: \n";
    std::cout << "conv1: " << t_conv1 << "\tus"  << '\n';
    std::cout << "pool1: " << t_pool1 << "\tus"  << '\n';
    std::cout << "conv2: " << t_conv2 << "\tus"  << '\n';
    std::cout << "pool2: " << t_pool2 << "\tus"  << '\n';
    std::cout << "fc1  : " << t_fc1   << "\tus"  << '\n';
    std::cout << "fc2  : " << t_fc2   << "\tus"  << '\n';
    std::cout << "soft : " << t_soft  << "\tus"  << '\n';
    std::cout << std::endl;

    std::size_t class_no = 0;
    std::cout << std::fixed << std::setprecision(3);
    for(auto c : class_out.buffer)
    {
        std::cout << "Number: " << class_no << "\t\t\t Confidence: %" << c * 100 << '\n';
        ++class_no;
    }
    std::cout << std::scientific << std::setprecision(6) << std::endl;

    return class_out;
}
