#include "ModelImporter.h"

#include "utils/nn_csv_reader.h"

ModelImporter::ModelImporter(const std::string & model) : model_file(model)
{
    auto all_buffers = read_csv_buffers<' ', '\n'>(model_file);
    
    for(auto & buffer : all_buffers)
    {
        DataBlob<float> data;
        data.buffer = buffer.buffer;
        data.dims = buffer.dims;
        buffers[buffer.name] = data;
    }
}

DataBlob<float> ModelImporter::get_buffer(const std::string & buf_name)
{
    return buffers[buf_name];
}
