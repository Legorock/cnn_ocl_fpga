#include "cnn_runall.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>


#include "seq_cnn.h"
#include "ocl_cnn.h"

#include "ModelImporter.h"
#include "mnist/mnist_reader_less.hpp"

#include "helper.h"
#include "Measure.h"

//const std::vector<std::string> kernel_names = {"softmax_layer", 
//                                               "max_pool1", "max_pool2", 
//                                               "conv1", "conv2", "fc1", "fc2"};
const std::vector<std::string> kernel_names = {"load_model_ocm", "softmax_layer", 
                                               "max_pool1", "max_pool2", 
                                               "conv1", "conv2", "fc1", "fc2"};

// Considers only images that are 28x28 like MNIST dataset images
void mnist_img_out(std::ostream& out, const std::vector<float>& img)
{
    std::size_t d = 28;
    for(std::size_t h = 0; h < d; ++h)
    {
        for(std::size_t w = 0; w < d; ++w)
        {
            if(img[w + h * d] > 0.0f)
                out << 1 << ' ';
            else
                out << 0 << ' ';
        }
        out << '\n';
    }
    out << std::endl;
}

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

std::uint8_t oneshot_to_label(const std::vector<float>& oneshot)
{
    auto max_it = std::max_element(oneshot.begin(), oneshot.end());
    return std::distance(oneshot.begin(), max_it);
}

std::map<std::string, Data> getModel(const std::string & model_path)
{
    std::cout << "Importing Model...\n";
    ModelImporter importer(model_path);
    std::cout << "Model is Imported!\n";
    return importer.get_buffers();
}

//float getAccuracy(const std::vector<std::uint8_t> & preds, const std::vector<std::uint8_t> & labels)
float getAccuracy(const std::vector<std::vector<float>> & preds, const std::vector<std::vector<float>> & labels)
{
    std::size_t num_correct = 0;
    if(preds.size() != labels.size()) 
    {
        std::cerr << "Num of predictions and number of labels do not match!!\n"
                  << "Terminating....\n";
        std::exit(EXIT_FAILURE);
    }
    for(std::size_t i = 0; i < preds.size(); ++i)
    {
        auto pl = oneshot_to_label(preds[i]);
        auto ll = oneshot_to_label(labels[i]);

        if(pl == ll) num_correct++;
    }
    return static_cast<float>(num_correct) / preds.size();
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
    std::cout << "ALL MNIST Run...\n";
    const std::size_t num_test = 3;

    std::vector<std::vector<float>> seq_class;
    std::vector<std::vector<float>> ocl_class;
    seq_class.reserve(num_test);
    ocl_class.reserve(num_test);
    
    auto partial_test_imgs = std::vector<std::vector<float>>(num_test);
    std::copy(test_imgs.begin(), test_imgs.begin()+num_test, partial_test_imgs.begin());
    auto partial_test_labels = std::vector<std::vector<float>>(num_test);
    std::copy(test_labels.begin(), test_labels.begin()+num_test, partial_test_labels.begin());

    oclCNN cnn_dev(model_params, m_world, kernels);

    for(auto & in_img : partial_test_imgs)
    {
        DataBlob<float> img = {in_img, {28, 28, 1}};

        auto seq_img_class = seq_cnn_img_test(img, model_params);
        seq_class.push_back(seq_img_class.buffer);
        std::cout << "Sequential Output: \n";
        print_classes(seq_img_class.buffer);

        auto ocl_img_class = cnn_dev.runImg(img);
        ocl_class.push_back(ocl_img_class.buffer);
        std::cout << "OCL FPGA Output: \n";
        print_classes(ocl_img_class.buffer);
    }
    std::cout << std::endl;

    std::cout << std::fixed << std::setprecision(4);
//    std::cout << "Sequential CPU run accuracy: " << getAccuracy(seq_class, test_labels) * 100 << "%" << std::endl;
//    std::cout << "OpenCL run accuracy: " << getAccuracy(ocl_class, test_labels) * 100 << "%" << std::endl;
    std::cout << "Sequential run accuracy: " << getAccuracy(seq_class, partial_test_labels) * 100 << "%" << std::endl;
    std::cout << "OpenCL run accuracy: " << getAccuracy(ocl_class, partial_test_labels) * 100 << "%" << std::endl;
    return 0.0f;
}

