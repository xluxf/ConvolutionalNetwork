//
// Created by tomas on 11.12.16.
//
#include "tools.h"


void parseWeights (std::string weights_str, std::vector<double> &weights){

    std::string value;
    std::size_t position;

    while ((position = weights_str.find(',')) != std::string::npos && weights_str.compare(",")) {
        value = weights_str.substr (0,position);
        weights_str = weights_str.substr (++position);
        weights.push_back((double)std::stof(value));
   }
}

void parseLogLine (std::string line,  int &layerCode, int &neurons, int &inputs, std::vector<double> &weights){

    std::string field, value, label;
    std::size_t position, position_f;

    // parse parameters
    while ((position = line.find('|')) != std::string::npos){

        field = line.substr (0,position);
        line = line.substr (++position);

        // parse field
        position_f = field.find(':');

        label = field.substr(0, position_f);
        value = field.substr(++position_f);

        if (!label.compare("layerCode")){
            layerCode = std::stoi(value);
        } else if (!label.compare("neurons")){
            neurons = std::stoi(value);
        }
    }

    // parse weights
    field = line;
    position_f = field.find(':');
    label = field.substr(0, position_f);
    value = field.substr(++position_f);
    parseWeights (value, weights);

}

void read(std::string filename, long n, std::vector<double> &r, std::vector<double> &g, std::vector<double> &b)
{
    // open file
    ifstream file (filename.c_str(), ios::in | ios::binary);

    int DIM_SQR = 1024;

    char label;
    char red[DIM_SQR];
    char green[DIM_SQR];
    char blue[DIM_SQR];


    // read
    while (!file.eof())
    {
        // get label and each of these channels
        file.get(label);
        file.read(red, DIM_SQR);
        file.read(green,DIM_SQR);
        file.read(blue, DIM_SQR);

    }


    file.close();

    return 0;
}



