//
// Created by Filip Lux on 24.11.16.
//

#ifndef NN_NEURON_FCLAYER_H
#define NN_NEURON_FCLAYER_H


/**
*@brief FullConnected layer
*/
class FCLayer {
public:
    /**
    *@brief number of neurons in this layer, in lower layer, in upper layer
    */
    int n, in, ou;
    /**
    *@brief pointer to upper layer, NULL if layer is output
    */
    FCLayer* up;
    /**
    *@brief pointer to lower layer, NULL if layer is input
    */
    FCLayer* down;
    /**
    *@brief output of neurons
    */
    double *out;
    /**
    *@brief ddot of neurons
    */
    double *ddot;
    /**
    *@brief weights of neurons
    */
    double *w;
    /**
    *@brief bias of layer
    */
    double bias;


    /**
    *@brief constructor for the first layer
    *@param inputs number of inputs
    *@param neurons number of neurons
    */
    FCLayer(int &inputs, int &neurons);

    /**
    *@brief constructor for the upper layers
    *@param inputs number of inputs
    *@param neurons number of neurons
    *@param lower lower layer
    */
    FCLayer(int &inputs, int &neurons, FCLayer* lower);

    /**
    *@brief forward
    */
    void forward_layer();
    /**
    * @brief backpropagation
    */
    void backProp_layer();
    /**
    * @brief backpropagation for last layer
    * @param result vector of expecting answer
    */
    void backProp_layer(std::vector <double> &result);
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
    void update_input(double* in);

    double* down_out;
    double* up_out;
    double* up_w;
    double* up_ddot;

    /**
    * @brief destructor
    */
    ~FCLayer();


};


#endif //NN_NEURON_FCLAYER_H
