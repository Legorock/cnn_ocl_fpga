#pragma once

#include <vector>
#include <string>
#include <map>

#include "DataBlob.h"

class ModelImporter
{
public:
    ModelImporter(const std::string & model);
    DataBlob<float> get_buffer(const std::string & buf_name);
    std::map<std::string, DataBlob<float>> get_buffers();
private:
    std::string model_file;
    std::map<std::string, DataBlob<float>> buffers;
};
