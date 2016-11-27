//
// Created by Filip Lux on 25.11.16.
//

#include "FCLayer.h"

#ifndef NN_NEURON_NEURALNET_H
#define NN_NEURON_NEURALNET_H

class NeuralNet {
public:
    //int deep;
    unsigned long width;
    FCLayer* input;
    FCLayer* output;

    NeuralNet(unsigned long d, unsigned long neurons) {
        //deep = d;
        width = neurons;
        input = new FCLayer(neurons,neurons);
        FCLayer* pointer = input;
        for (int i = 0; i< d-1; i++) {
            FCLayer* l = new FCLayer(neurons,neurons);
            pointer->up = l;
            l->down = pointer;
            pointer = l;
        }
        output = pointer;
    }

    void forward(std::vector <double> &in) {
        FCLayer* pointer = input;
        pointer->update(in);
        while (pointer != output) {
            pointer = pointer->up;
            pointer->forward_layer();
        }
    }

    void backProp(std::vector<double> &result) {
        FCLayer* pointer = output;
        pointer->backProp_layer(result);
        pointer = pointer->down;
        while (pointer != input) {
            pointer->backProp_layer();
            pointer = pointer->down;
        }
    }

    void print() {
        FCLayer* pointer = input;
        pointer->print();
        while (pointer != output) {
            pointer = pointer->up;
            pointer->print();
        }
    }
};

#endif //NN_NEURON_NEURALNET_H
