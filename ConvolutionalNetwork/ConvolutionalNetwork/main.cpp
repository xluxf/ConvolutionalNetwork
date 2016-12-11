#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#include "NeuralNet.h"
#include "read.cpp"

#define TRAINING_SET1 "data_batch_1.bin"
#define TRAINING_SET2 "data_batch_2.bin"
#define TRAINING_SET3 "data_batch_3.bin"
#define TRAINING_SET4 "data_batch_4.bin"
#define TEST_SET "test_batch.bin"


const int LAYERS = 3; // number of FC layers including output
const int NEURONS = 2; //number of neurons in one layer
//const int ITERATIONS = 10000; //number of learning cycles


int main(){

//    NeuralNet *net = new NeuralNet(LAYERS, NEURONS, 10, 10);

    char path [1024] = "cnn.log";

    NeuralNet *net = new NeuralNet(path);

//    net->network_save(path);


    return 0;
    
}

