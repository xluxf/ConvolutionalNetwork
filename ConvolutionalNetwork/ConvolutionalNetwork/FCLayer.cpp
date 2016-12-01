//
// Created by Filip Lux on 24.11.16.
//

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#include "FCLayer.h"

#define INIT_MAX 0.1  //max initialize weight
#define INIT_MIN -0.1 //min initialize weight
#define LR 0.002 //learning rate


//random function used for initialisation of weights
static double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


//activation function
static double sigma(double x) {
    return 1/(1+exp(-x));
}


// fullConnected layer
FCLayer::FCLayer(int &inputs, int &neurons, FCLayer* lower) { //creates layer, number of inputs and neurons
    n = neurons;
    in = inputs;
    down = lower;
    down_out = &down->out[0];
    down->ou = n;
    ddot = new double[n];
    out = new double[n];
    w = new double[in*n];
    bias = fRand(INIT_MIN,INIT_MAX);

    down->up_out = out;
    down->up_w = w;
    down->up_ddot = ddot;


    for (int i = 0; i < in*n; ++i) { //randomly initializes weights
        w[i] = fRand(INIT_MIN,INIT_MAX);
    }
}

FCLayer::FCLayer(int &inputs, int &neurons) {
    n = neurons;
    in = inputs;
    ddot = new double[n];
    out = new double[n];
    w = new double[in*n];
    bias = fRand(INIT_MIN,INIT_MAX);

    //down->up_out = out;
    //down->up_w = w;
    //down->up_ddot = ddot;


    for (int i = 0; i < in*n; ++i) { //randomly initializes weights
        w[i] = fRand(INIT_MIN,INIT_MAX);
    }

};

FCLayer::~FCLayer() {
    delete w;
    delete out;
    delete ddot;
};

void FCLayer::forward_layer() { //step forward with activation function
    for (int i = 0; i < n; i++) {
        out[i] = bias;
        for (int j = 0; j < in; j++) {
            out[i] += w[i*in+j] * down_out[j];
        }
        out[i] = sigma(out[i]);
    }
}

void FCLayer::backProp_layer() {
    for (int i = 0; i < n; i++) {
        ddot[i] = 0;
        for (int j = 0; j < ou; j++) {
           ddot[i] += up_ddot[j] * up_w[i+j*ou];
        }
    }
    FCLayer::learn();
}

void FCLayer::backProp_layer(std::vector <double> &result) {
    for (int i = 0; i < n; i++) {
        ddot[i] = (out[i] - result[i]) * out[i] * (1-out[i]);
    }
    FCLayer::learn();

}

void FCLayer::print() {
    std::cout << "layer weights:" << std::endl;
    for (int i = 0; i < n*n+1; i++) {
        std::cout <<  "w" << i << ": " << w[i] << ", ";
    }
    std::cout << std::endl;
}

void FCLayer::learn() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j<in; j++) {
            w[i*in+j] -= ddot[i] * down_out[j] * LR;
        }
    }
}

void FCLayer::update_input(double* in) {
    down_out = in;
};


