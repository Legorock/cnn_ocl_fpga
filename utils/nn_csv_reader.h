#pragma once

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


template<char delimeter, char endline>
std::vector<Buffer> read_csv_buffers(const std::string & filename)
{
  std::ifstream i(filename);
  std::vector<Buffer> buffers;

  int l = 0;
  for(std::string line; std::getline(i, line, i.widen(endline));)
  {
    if(l % 2 == 0)
    {
      Buffer b;
      std::stringstream ss;
      ss.str(line);
      std::string e;
      std::vector<std::string> strings;
      while(std::getline(ss, e, delimeter))
        strings.push_back(e);

      b.name = strings[1];
      for(std::size_t i = 2; i < strings.size(); ++i)
      {
        b.dims.push_back(std::stoi(strings[i]));
      }
      buffers.push_back(b);
    }
    else
    {
      auto & buf = buffers[buffers.size()-1];
      std::stringstream ss;
      ss.str(line);
      std::string e;
      auto n = std::count(line.begin(), line.end(), delimeter);
      std::vector<std::string> strings; strings.reserve(n);
      while(std::getline(ss, e, delimeter))
        strings.push_back(e);
      
      buf.buffer.resize(n);
      for(std::size_t i = 0; i < strings.size(); ++i)
      {
        buf.buffer[i] = std::stof(strings[i]);
      }
    }
    ++l;
  }
  return buffers;
}

