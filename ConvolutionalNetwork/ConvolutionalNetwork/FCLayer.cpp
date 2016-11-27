//
// Created by Filip Lux on 24.11.16.
//

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#include "FCLayer.h"

static const double INIT_MAX = 0.001;  //max initialize weight
static const double INIT_MIN = -0.001; //min initialize weight
static const double LR = 0.01; //learning rate

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


unsigned long n, in;
FCLayer* up;
FCLayer* down;
std::vector <double> w, ddot, out;

FCLayer::FCLayer(unsigned long &inputs, unsigned long &neurons) { //creates layer, number of inputs and neurons
    n = neurons;
    in = inputs;
    for (int i = 0; i < inputs*neurons+1; ++i) { //randomly initializes weights, w[0] is bias
        w.push_back(fRand(INIT_MIN,INIT_MAX));
    }
    ddot.assign(n,0);
    out.assign(n,0);

}

void FCLayer::forward_layer() { //step forward with activation function
    double* input = &down->out[0];
    for (int i = 0; i < n; i++) {
        out[i] = w[0];
        for (int j = 0; j < in; j++) {
            out[i] += w[i*in+j+1] * input[j];
        }
        out[i] = sigma(out[i]);
    }
}

void FCLayer::backProp_layer() {
    double* upDDot = &up->ddot[0];
    double* upW = &up->w[0];
    double* upOut = &up->out[0];
    for (int i = 0; i < n; i++) {
        ddot[i] = 0;
        for (int j = 0; j < in; j++) {
            ddot[i] += upDDot[j] * upW[1+i+j*in];
        }
        ddot[i] *= upOut[i] * (1-upOut[i]);
    }
    FCLayer::learn();
}

void FCLayer::backProp_layer(std::vector <double> result) {
    for (int i = 0; i < n; i++) {
        ddot[i] = (result[i] - out[i]) * out[i] * (1-out[i]);
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
    double* downOut = &down->out[0];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j<in; j++) {
            w[i*n + j + 1] += ddot[i] * downOut[j] * LR;
        }
    }
}

void FCLayer::update(std::vector <double> &in) {
    for (int i = 0; i<n; i++) {
        out[i] = in[i];
    }
};


