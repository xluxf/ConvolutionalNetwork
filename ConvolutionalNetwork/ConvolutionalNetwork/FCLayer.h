//
// Created by Filip Lux on 24.11.16.
//

#ifndef NN_NEURON_FCLAYER_H
#define NN_NEURON_FCLAYER_H

#include <vector>

/**
*@brief FullConnected layer
*/
class FCLayer {
public:
    /**
    *@brief number of neurons, number of inputs
    */
    unsigned long n, in;
    /**
    *@brief pointer to upper layer, NULL if layer is output
    */
    FCLayer* up;
    /**
    *@brief pointer to lower layer, NULL if layer is input
    */
    FCLayer* down;
    /**
    *@brief weights, ddot of neurons, output od neurons
    */
    std::vector <double> w, ddot, out;


    /**
    *@brief constructor
    *@param inputs number of inputs
    *@param neurons number of neurons
    */
    FCLayer(unsigned long &inputs, unsigned long &neurons);

    /**
    *@brief forward
    */
    void forward_layer();
    /**
    * @brief backpropagation
    */
    void backProp_layer();
    /**
    * @brief backpropagation of output layer
    * @param result vector of expecting answer
    */
    void backProp_layer(std::vector <double> result);
    /**
    * @brief print weights
    */
    void print();
    /**
    * @brief change weights
    */
    void learn();
    /**
    * @brief unsert new values to input
    */
    void update(std::vector <double> &in);
};


#endif //NN_NEURON_FCLAYER_H
