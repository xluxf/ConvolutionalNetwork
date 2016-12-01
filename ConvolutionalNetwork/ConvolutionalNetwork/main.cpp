#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#include "NeuralNet.h"
#include "read.cpp"

//
#define TRAINING_SET1 "data_batch_1.bin"
#define TRAINING_SET2 "data_batch_2.bin"
#define TEST_SET "test_batch.bin"


const int LAYERS = 2; // number of FC layers including output
//const int NEURONS = 50; //number of neurons in one layer
//const int ITERATIONS = 10000; //number of learning cycles


int main() {
    NeuralNet *net = new NeuralNet(LAYERS, DIM_SQR, DIM_SQR, 10);
    std::vector<double> in, result;

    string filename = TRAINING_SET1;

    // open file
    ifstream file (filename.c_str(), ios::in | ios::binary);

    if (!file.is_open())
    {
        cout << "Error opening files" << endl;
        return -1;
    }

    char label;
    char red[DIM_SQR];
    char green[DIM_SQR];
    char blue[DIM_SQR];

    int right = 0;
    // read
    for (int i = 1; i <= 10000; i++)
    {
        // get label and each of these channels
        file.get(label);
        file.read(red, DIM_SQR);
        file.read(green,DIM_SQR);
        file.read(blue, DIM_SQR);
        net->network_forward(red,green,blue);
        if (net->network_check(label)) right++; //counts correst samples
        net->network_backprop(static_cast<int>(label));

        if (i  % 1000 == 0) {   //prints how many samples was correct
            std::cout << i << ". correct: " << right << std::endl;
            right = 0;
        }
#if 0 //print all outputs, ddots, weights, ...
        std::cout << "step " << i << " :\n";
        std::cout << "out: ";
        for (int j = 0; j < 10; j++) {
            std::cout << net -> output[j]<< ", ";
        }
        std::cout << endl;
        std::cout << "ddot: ";
        for (int j = 0; j < 10; j++) {
            std::cout << net -> last -> ddot[j] << ", ";
        }
        std::cout << endl;
        std::cout << "w0: ";
        for (int j = 1; j <= 10; j++) {
            std::cout << net -> last -> w[j] << ", ";
        }
        std::cout << "... , ";
        for (int j = 1020; j <= 1023; j++) {
            std::cout << net -> last -> w[j] << ", ";
        }
        std::cout << "\nw1: ";
        for (int j = 1024; j <= 1034; j++) {
            std::cout << net -> last -> w[j] << ", ";
        }
        std::cout << "... , ";
        for (int j = 2045; j <= 2047; j++) {
            std::cout << net -> last -> w[j] << ", ";
        }
        std::cout << "\nw2: ";
        for (int j = 2048; j <= 2058; j++) {
            std::cout << net -> last -> w[j] << ", ";
        }
        std::cout << "...\n";
        std::cout << "w3: ";
        for (int j = 3072; j <= 3082; j++) {
            std::cout << net -> last -> w[j] << ", ";
        }
        std::cout << "...\n";
#endif
    }
    file.close();

#if 1 //2. train_batch
    // 2. training
    filename = TRAINING_SET2;

    // open file
    ifstream file2 (filename.c_str(), ios::in | ios::binary);

    if (!file2.is_open())
    {
        cout << "Error opening file" << endl;
        return -1;
    }

    for (int i = 10001; i <= 20000; i++)
    {
        // get label and each of these channels
        file2.get(label);
        file2.read(red, DIM_SQR);
        file2.read(green,DIM_SQR);
        file2.read(blue, DIM_SQR);
        net->network_forward(red,green,blue);
        if (net->network_check(label)) right++;
        net->network_backprop(static_cast<int>(label));

        if (i % 1000 == 0) {
            std::cout << i << ". correct: " << right << std::endl;
            right = 0;
        }

    }
    file2.close();

#endif

    string test_filename = TEST_SET;

    // open file
    ifstream test_file (test_filename.c_str(), ios::in | ios::binary);

    if (!test_file.is_open())
    {
        cout << "Error opening file" << endl;
        return -1;
    }

    right = 0;
    int ans[10] = {};
    for (int i = 0; i < 1000; i++)
    //while (!test_file.eof())
    {
        // get label and channels
        test_file.get(label);
        test_file.read(red, DIM_SQR);
        test_file.read(green,DIM_SQR);
        test_file.read(blue, DIM_SQR);
        net->network_forward(red,green,blue);
        if (net->network_check(label)) {
            ++right;
            ++ans[label];
            std::cout << right << ". spravna odpoved je " << static_cast<int>(label) << std::endl;

            for (int j = 0; j < 10; j++) {
                std::cout << j << ": " <<net -> output[j]<< ", ";
            }
            std::cout << endl;
        }
    }

    std::cout << "pocty spravnych odpovedi v kategoriích: ";
    for (int j = 0; j < 10; j++) {
        std::cout << ans[j] << ", ";
    }
    std::cout << endl;

    test_file.close();
    std::cout << "spravnost: " << right*100/1000 << "%" <<std::endl;

    return 0;
}