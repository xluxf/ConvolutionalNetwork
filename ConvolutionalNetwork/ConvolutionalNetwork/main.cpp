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
//const int NEURONS = 50; //number of neurons in one layer
//const int ITERATIONS = 10000; //number of learning cycles

#if 0
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

#if 1 //3. train_batch
    // 3. training
    filename = TRAINING_SET3;

    // open file
    ifstream file3 (filename.c_str(), ios::in | ios::binary);

    if (!file3.is_open())
    {
        cout << "Error opening file" << endl;
        return -1;
    }

    for (int i = 20001; i <= 30000; i++)
    {
        // get label and each of these channels
        file3.get(label);
        file3.read(red, DIM_SQR);
        file3.read(green,DIM_SQR);
        file3.read(blue, DIM_SQR);
        net->network_forward(red,green,blue);
        if (net->network_check(label)) right++;
        net->network_backprop(static_cast<int>(label));

        if (i % 1000 == 0) {
            std::cout << i << ". correct: " << right << std::endl;
            right = 0;
        }

    }
    file3.close();

#endif

#if 1 //4. train_batch
    // 4. training
    filename = TRAINING_SET4;

    // open file
    ifstream file4 (filename.c_str(), ios::in | ios::binary);

    if (!file4.is_open())
    {
        cout << "Error opening file" << endl;
        return -1;
    }

    for (int i = 30001; i <= 40000; i++)
    {
        // get label and each of these channels
        file4.get(label);
        file4.read(red, DIM_SQR);
        file4.read(green,DIM_SQR);
        file4.read(blue, DIM_SQR);
        net->network_forward(red,green,blue);
        if (net->network_check(label)) right++;
        net->network_backprop(static_cast<int>(label));

        if (i % 1000 == 0) {
            std::cout << i << ". correct: " << right << std::endl;
            right = 0;
        }

    }
    file4.close();

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

#endif


#if 1           //test forward ConvLayer
double dotProduct(const int* a,  const int* b, int dim_a, int dim_b, int steps_i, int steps_j) {
    double ans = 0;
    for (int i = 0; i <= steps_i; i++) {
        for (int j = 0; j <= steps_j; j++) {
            ans += a[i*dim_a +j] * b[i*dim_b + j];
        }
    }
    return ans;
}


int main() {




    const int w_dim = 3;
    const int depth = 2;
    const int wn = w_dim * w_dim;
    const int input_depth = 3;
    const int dim = 5;
    const int n = dim * dim;
    const int bias[4] = {1,0,1,0};
    int out[n*depth] = {};
    const int input[3*n] =
                       {2,1,2,1,1,
                       2,2,2,1,0,
                       0,0,0,0,1,
                       2,0,2,1,0,
                       2,2,0,0,2,

                       2,1,0,0,2,
                       2,2,0,0,2,
                       0,0,1,2,1,
                       1,2,1,2,0,
                       2,0,2,1,0,

                       1,1,2,1,0,
                       2,0,1,1,1,
                       0,1,2,2,1,
                       1,0,1,2,0,
                       2,0,1,0,1};

    const int w[w_dim*w_dim*input_depth* depth] = {
            -1,1,-1,
            1,1,-1,
            1,0,0,

            -1,1,0,
            1,-1,0,
            0,1,-1,

            -1,1,1,
            -1,1,0,
             0,1,0,

            1,0,0,
            0,-1,-1,
            1,1,0,

            1,1,1,
            0,-1,-1,
            -1,0,0,

            0,1,0,
            0,-1,0,
            1,1,0
    };


    int diff = w_dim/2;
    //for (int i = 0; i < n; i++) out[i] = bias[i];

    // for every location of every filter on the input
    // every possible  out += w x down_out

    for (int f = 0; f < depth; f++) {       //number of the filter, number of out layer
        int pf = f * wn * input_depth;          //position of the start of the filter
        int of = f*n;            //position of the start of the out
        for (int i = 0; i < dim; i++) {        //x coordinates
            for (int j = 0; j < dim; j++) {        //y coordinates
                int p = i*dim + j;           //field of out
                out[of + p] = bias[f];         //set out to bias

                int ii = std::max(0, i-diff);
                int jj = std::max(0, j-diff);
                int di = std::min(dim-1, i+diff) - ii;
                int dj = std::min(dim-1, j+diff) - jj;

                //urceni zacatku nasobeni

                for (int filter_layer = 0; filter_layer < input_depth; filter_layer++)
                    out[of + p] += dotProduct(&input[filter_layer * n + ii * dim + jj],
                                              &w[pf + wn * filter_layer + (ii-i+diff) * w_dim + (jj-j+diff)],
                                              dim, w_dim, di, dj);


            }
        }
    }

    for (int i = 0; i < 3*9; i++) {
        std::cout << w[i] <<", ";
        if ((i+1) % 3 == 0) std::cout << '\n';
        if ((i+1) % 9 == 0) std::cout << '\n';
    }

    for (int i = 0; i < 3*n; i++) {
        std::cout << input[i] <<", ";
        if ((i+1) % 5 == 0) std::cout << '\n';
        if ((i+1) % 25 == 0) std::cout << '\n';
    }


    for (int i = 0; i < 2*n; i++) {
        std::cout << out[i] <<", ";
        if ((i+1) % 5 == 0) std::cout << '\n';
        if ((i+1) % 25 == 0) std::cout << '\n';
    }


    return 0;
}

#endif