#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

const double INIT_MAX = 0.001;
const double INIT_MIN = -0.001;
const double LR = 0.01;

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


//activation function
double sigma(double &x) {
    return 1/(1+exp(-x));
}


// fullconnected layer
class FCLayer {
    public:
    int n, in;
    FCLayer* up;
    FCLayer* down;
    std::vector <double> w, ddot, out;

    FCLayer(int &inputs, int &neurons) { //creates layer, number of inputs and neurons
        n = neurons;
        in = inputs;
        for (int i = 0; i < inputs*neurons+1; ++i) { //randomly initializes weights, w[0] is bias
            w.push_back(fRand(INIT_MIN,INIT_MAX));
        }
        ddot.assign(n,0);
        out.assign(n,0);

    }

    void forward_layer() { //step forward with activation function
        double* input = &down->out[0];
        for (int i = 0; i < n; i++) {
            out[i] = w[0];
            for (int j = 0; j < in; j++) {
                out[i] += w[i*in+j+1] * input[j];
            }
            out[i] = sigma(out[i]);
        }
    }

    double backProp_layer() {
        double* upddot = &up->ddot[0];
        double* upw = &up->w[0];
        double* upout = &up->out[0];
        for (int i = 0; i < n; i++) {
            ddot[i] = 0;
            for (int j = 0; j < in; j++) {
                ddot[i] += upddot[j] * upw[1+i+j*in];
            }
            ddot[i] *= upout[i] * (1-upout[i]);
        }
        learn();
    }

    double backProp_layer(std::vector <double> result) {
        for (int i = 0; i < n; i++) {
            ddot[i] = (result[i] - out[i]) * out[i] * (1-out[i]);
        }
        learn();

    }

    void print() {
        std::cout << "layer weights:" << std::endl;
        for (int i = 0; i < n*n+1; i++) {
            std::cout <<  "w" << i << ": " << w[i] << ", ";
        }
        std::cout << std::endl;
    }

    void learn() {
        double* downout = &down->out[0];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j<in; j++) {
                w[i*n + j + 1] += ddot[i] * downout[j] * LR;
            }
        }

    }

    void update(std::vector <double> &in) {
        for (int i = 0; i<n; i++) {
            out[i] = in[i];
        }
    }
};


class NeuralNet {
    public:
    int deep;
    int width;
    FCLayer* input;
    FCLayer* output;

    NeuralNet(int d, int neurons) {
        deep = d;
        width = neurons;
        input = new FCLayer(d,neurons);
        FCLayer* pointer = input;
        for (int i = 0; i< d-1; i++) {
            FCLayer* l = new FCLayer(neurons,neurons);
            pointer->up = l;
            l->down = pointer;
            pointer = l;
        }
        output = pointer;
    }

    void forward(std::vector <double> &in) {
        FCLayer* pointer = input;
        pointer->update(in);
        while (pointer != output) {
            pointer = pointer->up;
            pointer->forward_layer();
        }
    }

    void backProp(std::vector<double> &result) {
        FCLayer* pointer = output;
        pointer->backProp_layer(result);
        pointer = pointer->down;
        while (pointer != input) {
            pointer->backProp_layer();
            pointer = pointer->down;
        }
    }

    void print() {
        FCLayer* pointer = input;
        pointer->print();
        while (pointer != output) {
            pointer = pointer->up;
            pointer->print();
        }
    }
};

const int LAYERS = 10; // number of FC layers including output and input layer
const int NEURONS = 4; //number of neurons in one layer
const int ITERATIONS = 80000; //number of learning cycles


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
    in.push_back(0.05);
    in.push_back(0.1);
    in.push_back(0.2);
    in.push_back(0.5);

    result.push_back(0.1);
    result.push_back(0.1);
    result.push_back(0.1);
    result.push_back(0.9);

    net->print();


    for (int j = 0; j < ITERATIONS; j++) {
        net->forward(in);

        double error = 0.0;  //
        for (int i = 0; i < net->width; i++) {
            error += pow(net->output->out[i] - result[i], 2) / 2;
            std::cout << "x[" << i << "] = " << net->output->out[i] << ", ";
        }
        std::cout << std::endl << "error = " << error << std::endl;
        std::cout << pow( net->output->out[0] - result[0], 2) << ", "<<  pow(net->output->out[1] - result[1], 2) << std::endl;


        net->backProp(result);
        //net->print();
    }
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