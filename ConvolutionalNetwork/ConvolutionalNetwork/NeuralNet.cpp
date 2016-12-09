//
// Created by Filip Lux on 30.11.16.
//

#include <vector>
#include "FCLayer.h"
#include "NeuralNet.h"
#include <memory>
#include <iostream>
#include <fstream>

NeuralNet::NeuralNet(int deep, int neurons, int input_size, int output_size) {
    NeuralNet::input_size = input_size;
    NeuralNet::output_size = output_size;

    first = new FCLayer(input_size,neurons);
    Layer* pointer = first;
    for (int i = 1; i< deep - 1; i++) {
        pointer->up = new FCLayer(neurons, neurons, pointer);
        pointer = pointer->up;
    }
    pointer->up = new FCLayer(neurons, output_size, pointer);
    pointer = pointer->up;
    last = pointer;
    output = &last->out[0];
};

void NeuralNet::network_updateInput(char* r, char* g, char* b){
    input[0] = r;
    input[1] = g;
    input[2] = b;
};

void NeuralNet::network_backprop(int l){
    answer = l;
    std::vector <double> result;
    result.assign(10,0.01);
    result[answer] = 0.99;
    backProp(result);
};

void NeuralNet::network_forward(char* r, char* g, char* b){
    network_updateInput(r,g,b);
    forward();
};

bool NeuralNet::network_check(char &label) {
    for (int i = 0; i < output_size; i++) {
        if (output[i] > output[label]) {
            return false;
        }
    }
    return true;
};

void NeuralNet::forward() {

    double in[input_size];
    for (unsigned int i = 0; i < input_size; i++)
        in[i] = static_cast<int>(input[0][i]);
    first->update_input(&in[0]);

    Layer* pointer = first;
    pointer->forward_layer();
    while (pointer != last) {
        pointer = pointer->up;
        pointer->forward_layer();
    }
};

void NeuralNet::backProp(std::vector<double> &result) {

    last->backProp_layer(result);

    Layer* pointer = last->down;
    while (pointer != first) {
        pointer->backProp_layer();
        pointer = pointer->down;
    }
};

void NeuralNet::print() {
    Layer* pointer = first;
    pointer->print();
    while (pointer != last) {
        pointer = pointer->up;
        pointer->print();
    }
};

void NeuralNet::network_save(char* path) {
    
    std::ofstream logfile (path);
    double* up_w;
    int n, in;
    
    Layer* layer = first;
  
    while (layer != last) {
        
        logfile << layer->getType() << ": ";
        
        up_w = layer->up_w; 
        in = layer->in;
        n = layer->n;
        
        
        for (int i=0; i < in * n; i++)
            logfile << up_w[i] << ", ";
                
        logfile << "\n";
        layer = layer->up;
    }
    
    logfile.close();
    
};


