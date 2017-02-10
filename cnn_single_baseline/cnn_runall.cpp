#include "cnn_runall.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include "ModelImporter.h"
#include "mnist/mnist_reader_less.hpp"

#include "helper.h"
#include "Measure.h"

const std::vector<std::string> kernel_names = {"max_pool2", "conv_local", "softmax_layer", "fc"};

std::vector<float> imgcast_to_float(const std::vector<std::uint8_t>& img)
{
    std::vector<float> fimg(img.size());
    std::transform(img.begin(), img.end(), fimg.begin(), 
            [](std::uint8_t p)
            {
                return static_cast<float>(p) / 255.0f;
            }); 
    return fimg;
}

std::vector<float> label_to_oneshot(const std::uint8_t label)
{
    std::vector<float> oneshot(10, 0.0f);
    oneshot[label] = 1.0f;
    return oneshot;
}

std::map<std::string, Data> getModel(const std::string & model_path)
{
    std::cout << "Importing Model...\n";
    ModelImporter importer(model_path);
    std::cout << "Model is Imported!\n";
    return importer.get_buffers();

//    std::map<std::string, Data> model;
//    model["wc1"] = importer.get_buffer("wc1"); // Conv1 weights
//    model["bc1"] = importer.get_buffer("bc1"); // Conv1 biases
//    model["wc2"] = importer.get_buffer("wc2"); // Conv2 weights
//    model["bc2"] = importer.get_buffer("bc2"); // Conv2 biases
//    model["wd1"] = importer.get_buffer("wd1"); // FullyConn. (dense) weights
//    model["bd1"] = importer.get_buffer("bd1"); // FullyConn. (dense) biases
//    model["wdo"] = importer.get_buffer("wdo"); // FullyConn. out (dense) weights
//    model["bdo"] = importer.get_buffer("bdo"); // FullyConn. out (dense) biases
//    return model;
}

// Constructor
cnn_runall::cnn_runall(xcl_world & world, const char * clFilename, bool isBinary)
    : m_world(world), model_params(getModel("../lenet_data/model.csv"))
{
    std::cout << "Initializing OpenCL Kernels(Binary or Source)...\n";
    if(isBinary)
    {
        kernels = get_kernels_binary(world, clFilename, kernel_names);
    }
    else
    {
        kernels.reserve(kernel_names.size());
        for(std::size_t i = 0; i < kernel_names.size(); ++i)
        {
            kernels.push_back(xcl_import_source(world, clFilename, kernel_names[i].c_str()));
        }
    }
    std::cout << "OpenCL Kernels Initialized!\n";

    std::cout << "MNIST Dataset is loading....\n";
    mnist::mnist_path = "../mnist/";
    auto dataset = mnist::read_dataset<>();

    train_imgs.reserve(dataset.training_images.size());
    train_labels.reserve(dataset.training_labels.size());
    test_imgs.reserve(dataset.test_images.size());
    test_labels.reserve(dataset.test_labels.size());

    std::for_each(dataset.training_labels.begin(), dataset.training_labels.end(), 
            [&train_labels](const std::uint8_t l){ train_labels.push_back(label_to_oneshot(l)); });
    std::for_each(dataset.test_labels.begin(), dataset.test_labels.end(),
            [&test_labels](const std::uint8_t l){ test_labels.push_back(label_to_oneshot(l)); });

    std::for_each(dataset.training_images.begin(), dataset.training_images.end(),
            [&train_imgs](const std::vector<std::uint8_t>& p){ train_imgs.push_back(imgcast_to_float(p)); });
    std::for_each(dataset.test_images.begin(), dataset.test_images.end(),
            [&test_imgs](const std::vector<std::uint8_t>& p){ test_imgs.push_back(imgcast_to_float(p)); });
    std::cout << "MNIST Dataset is loaded!" << std::endl;
}

// Destructor
cnn_runall::~cnn_runall()
{
    for(std::size_t i = 0; i < kernels.size(); ++i)
    {
        clReleaseKernel(kernels[i]);
    }
}

float cnn_runall::run_all()
{
    return 0.0f;
}

//Data cnn_runall::seq_run_img(const Data& img)
//{
//    Data d;
//    return d;
//}
//
//Data cnn_runall::ocl_run_img(Data& img)
//{
//    Data d;
//    return d;
//}


