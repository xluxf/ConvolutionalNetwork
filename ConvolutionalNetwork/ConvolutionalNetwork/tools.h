//
// Created by tomas on 11.12.16.
//

#ifndef FULLCONNECTEDNEURALNET_TOOLS_H
#define FULLCONNECTEDNEURALNET_TOOLS_H

#include <iostream>
#include <vector>

void parseWeights (std::string weights_str, std::vector<double> &weights);
void parseLogLine (std::string line,  int &layerCode, int &neurons, int &inputs, std::vector<double> &weights);
void read(std::string filename, long n, std::vector<double> &r, std::vector<double> &g, std::vector<double> &b);

#endif //FULLCONNECTEDNEURALNET_TOOLS_H
