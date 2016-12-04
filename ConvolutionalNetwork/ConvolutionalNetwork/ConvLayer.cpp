//
// Created by Filip Lux on 27.11.16.
//

#include "ConvLayer.h"


double dotProduct(double* a,  double* b, int &dim_a, int &dim_b, const int &steps_i,const int &steps_j) {
    double ans = 0;
    for (int i = 0; i <= steps_i; i++) {
        for (int j = 0; j <= steps_j; j++) {
            ans += a[i*dim_a +j] * b[i*dim_b + j];
        }
    }
    return ans;
}

ConvLayer::ConvLayer(int filter_dim, int stroke, int filters) {

    //parameters of the input
    input_depth = down->depth;      //TODO: set input parameters
    input_dim = down->dim;
    input = down->out;

    //parameters of the output
    dim = input_dim;
    depth = filters;
    n = dim * dim;

    //parameters of the filters
    w_dim = filter_dim;
    wn = w_dim * w_dim;          //number of weights in one layer
    s = stroke;


    w = new double[depth*wn*input_depth];
    out = new double[depth*n];
    ddot = new double[depth*n];
    bias = new double[depth];

    for (int i = 0; i < in*n; ++i) { //randomly initializes weights
        w[i] = fRand(INIT_MIN,INIT_MAX);
    }
    for (int i = 0; i < n; ++i) { //randomly initializes weights
        bias[i] = fRand(INIT_MIN,INIT_MAX);
    }
}

ConvLayer::ConvLayer(int filter_dim, int stroke, int filters, Layer* lower) {
    //conection to lower layer
    down = lower;

    //parameters of the input
    input_depth = down->depth;
    input_dim = down->dim;
    input = down->out;

    //parameters of the output
    dim = input_dim;
    depth = filters;
    n = dim * dim;

    //parameters of the filters
    w_dim = filter_dim;
    wn = w_dim * w_dim;          //number of weights in one layer
    s = stroke;


    w = new double[depth*wn*input_depth];
    out = new double[depth*n];
    ddot = new double[depth*n];
    bias = new double[depth];

    for (int i = 0; i < in*n; ++i) { //randomly initializes weights
        w[i] = fRand(INIT_MIN,INIT_MAX);
    }
    for (int i = 0; i < n; ++i) { //randomly initializes weights
        bias[i] = fRand(INIT_MIN,INIT_MAX);
    }
}



void ConvLayer::forward_layer() {

    const int diff = w_dim/2;

    // for every output


    for (int f = 0; f < depth; f++) {       //number of the filter, number of out layer
        const int pf = f * wn * input_depth;          //position of the start of the filter
        const int of = f * n;                           //position of the start of the out
        for (int i = 0; i < dim; i++) {             //x coordinates
            for (int j = 0; j < dim; j++) {        //y coordinates
                int p = of + i*dim + j;           //field of out
                out[p] = bias[f];

                const int ii = std::max(0, i-diff);
                const int jj = std::max(0, j-diff);
                const int di = std::min(dim-1, i + diff) - ii;
                const int dj = std::min(dim-1, j + diff) - jj;

                for (int layer = 0; layer < input_depth; layer++)
                    out[p] += dotProduct(&input[layer * n + ii * dim + jj],
                                              &w[pf + wn * layer + (ii - i + diff)* w_dim + (jj-j+diff)],
                                              dim, w_dim, di, dj);

            }
        }
    }

}

void ConvLayer::backProp_layer(std::vector <double> result) {

}

void ConvLayer::backProp_layer() {

}

void ConvLayer::learn() {

}

void ConvLayer::update_input(double* in) {
    input = in;
}

ConvLayer::~ConvLayer() {
    delete bias;
    delete ddot;
    delete out;
    delete w;
};