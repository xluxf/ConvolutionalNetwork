#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#include "NeuralNet.h"


const int LAYERS = 3; // number of FC layers including output and input layer
const int NEURONS = 50; //number of neurons in one layer
const int ITERATIONS = 10000; //number of learning cycles


int main() {
    NeuralNet* net = new NeuralNet(LAYERS,NEURONS);
    std::vector <double> in, result;
    /**
    for (int i = 0; i<2;i++) {
        double s = fRand(1.0,10.0);
        result += s;
        in.push_back(s);
    }
     */

#if 1 //learning one vector


    in.assign(NEURONS-1, 0.3);
    in.push_back(0.9);
    result.assign(NEURONS-1, 0.1);
    result.push_back(0.9);


    //net->print();


    for (int j = 0; j < ITERATIONS; j++) {
        net->forward(in);

        /**
        double error = 0.0;  //
        for (int i = 0; i < net->width; i++) {
            error += pow(net->output->out[i] - result[i], 2) / 2;
            std::cout << "x[" << i << "] = " << net->output->out[i] << ", ";
        }
        std::cout << std::endl << "error = " << error << std::endl;
        std::cout << pow( net->output->out[0] - result[0], 2) << ", "<<  pow(net->output->out[1] - result[1], 2) << std::endl;
        */

        net->backProp(result);
    }


    double error = 0.0;  //
    for (int i = 0; i < net->width; i++) {
        error += pow(net->output->out[i] - result[i], 2) / 2;
        std::cout << "x[" << i << "] = " << net->output->out[i] << ", ";
    }

    std::cout << std::endl << "error = " << error << std::endl;
    //net->print();

    return 0;
#endif

#if 0 //try to learn some random values, it learns just probability distribution
    bool run = true;
    long cyc = 0;
    in.assign(4,0);
    result.assign(4,0);

    while (run) {
        for (int i = 0; i < NEURONS; i++) {
            in[i] = fRand(0.01,0.99);
        }
        net->forward(in);

        if (in[0] + in[1] + in[2] + in[3] > 3) {result [0] == 0.99; result [1] == 0.01;}
        else {result [0] == 0.1; result [1] == 0.99;}

        if (cyc == 10000) {
            cyc = 0;
            char ans;
            double error = 0.0;  //
            for (int i = 0; i < NEURONS; i++) {
                error += pow(net->output->out[i] - in[i], 2) / 2;
                std::cout << "x[" << i << "] = " << net->output->out[i] << ", ";
            }
            std::cout << std::endl;

            for (int i = 0; i < NEURONS; i++) {
                std::cout << "in[" << i <<"] = " << in[i] << ", ";
            }
            std::cout << std::endl;

            std::cin >> ans;
            if (ans == '0') run = false;
        }
        net->backProp(result);

        cyc++;
    }
    net->print();


#endif


}