#include "ModelImporter.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

struct Buffer
{
  std::vector<float> buffer;
  std::vector<std::size_t> dims;
  std::string name;
};

template<char delimeter>
std::vector<std::string> split_string(const std::string & s)
{
    std::istringstream iss(s);
    std::vector<std::string> strings;
    strings.reserve(std::count(s.begin(), s.end(), delimeter));
    std::string split;
    while(std::getline(iss, split, delimeter))
    {
        strings.push_back(split);
    }
    return strings;
}

template<char delimeter, char endline>
std::vector<Buffer> read_csv_buffers(const std::string & filename)
{
    std::ifstream i(filename);
    std::vector<Buffer> buffers;

    std::vector<std::vector<std::string>> csv;
    for(std::string line; std::getline(i, line, endline);)
    {
        auto split_str = split_string<delimeter>(line);
        csv.push_back(split_str);        
    }

    if(csv.size() % 2 != 0)  // Something wrong with the model file
    {
        std::cerr << "Number of lines in the model file are not multiple of 2, something is wrong; Terminating!\n";
        std::exit(EXIT_FAILURE);
    }

    buffers.reserve(csv.size()/2);

    for(std::size_t i = 0; i < csv.size(); i+=2)
    {
     // Debugging
     //   for(auto & s : (csv[i]))
     //   {
     //       std::cout << s << "\t\t";
     //   }
     //   std::cout << '\n';
        Buffer buf;
        buf.name = csv[i][1];
        buf.buffer.reserve(csv[i+1].size());
        for(std::size_t d = 2; d < csv[i].size(); ++d)
        {
            buf.dims.push_back(static_cast<std::size_t>(std::stoi(csv[i][d])));
        }

        for(std::size_t f = 0; f < csv[i+1].size(); ++f)
        {
            try{
            buf.buffer.push_back(static_cast<float>(std::stod(csv[i+1][f])));
            }
            catch(...)
            {
                std::cerr << "failed at: " << i+1 << '\t' << f << '\n';
                std::cerr << "val: " << csv[i+1][f] << '\n';
                std::exit(EXIT_FAILURE);
            }
        }
        buffers.push_back(buf);
    }
    return buffers;
}

ModelImporter::ModelImporter(const std::string & model) : model_file(model)
{
    auto all_buffers = read_csv_buffers<' ', '\n'>(model_file);
    std::cout << "Model File: " << model_file << " imported!\n"; 
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
    return buffers.at(buf_name);
}
