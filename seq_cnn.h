#pragma once

#include <string>
#include <map>

#include "seq.h"
#include "DataBlob.h"


DataBlob<float> seq_cnn_img_test(const DataBlob<float>& img, const std::map<std::string, DataBlob<float>>& model);

