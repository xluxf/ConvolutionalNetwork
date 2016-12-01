//
// Created by Filip Lux on 25.11.16.
//

#include "FCLayer.h"

#ifndef NN_NEURON_NEURALNET_H
#define NN_NEURON_NEURALNET_H



class NeuralNet {
public:
    /**
    *@brief inputs
    */
    char* red;
    char* green;
    char* blue;

    double* output;

    int answer, input_size, output_size;
    FCLayer* first;
    FCLayer* last;

    /**
    *@brief constructor for the full connected neural net
    *@param deep deep of the net
    *@param neurons number of neurons in one hidden layer
    *@param input_size size of input
    *@param output_size size of output
    */
    NeuralNet(int deep, int neurons, int input_size, int output_size) ;

    /**
    *@brief updates input
    *@param r red channel
    *@param g green channel
    *@param b blue channel
    */
    void network_updateInput(char* r, char* g, char* b);

    /**
    *@brief back propagation of network
    *@param l right class of the sample
    */
    void network_backprop(int l);

    /**
    *@brief step forward
    *@param r red channel
    *@param g green channel
    *@param b blue channel
    */
    void network_forward(char* r, char* g, char* b);

    /**
    *@brief check if answer is right
    *@param label right class of the sample
    */
    bool network_check(char &label);

    /**
    *@brief forward step without changing input
    */
    void forward() ;

    /**
    *@brief forward step without changing input
    *@param result vector of expected outputs
    */
    void backProp(std::vector<double> &result);

    /**
    *@brief prints all weights
    */
    void print();
};

#endif //NN_NEURON_NEURALNET_H
