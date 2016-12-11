#include "MNeuralNet.h"
#include "FCLayer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include <iostream>
#include <fstream>

using namespace std;

static const int NUMBER_OF_NEURONS = 10;

int readFile(string filename, Input* input) {
	
	ifstream file(filename.c_str(), ios::in | ios::binary);

	if (!file.is_open())
	{
		cout << "Error opening file" << endl;
		return -1;
	}

	// read
	while (!file.eof())
	{
		// get label and each of these channels
		file.get(input->label);
		file.read(input->red, DIM_SQR);
		file.read(input->green, DIM_SQR);
		file.read(input->blue, DIM_SQR);
	}
}

void convertFormat(MyNeuralNet* net) {
	const int inputSize = DIM_SQR * 3;

	double input[inputSize];
	for (int i = 0; i < DIM_SQR;i++) {
		//input[i] = net->input->red;
	}
	for (int i = DIM_SQR; i < DIM_SQR; i++) {
		//input[i] = net->input->red;
	}
	for (int i = DIM_SQR*2; i < DIM_SQR; i++) {
		//input[i] = net->input->red;
	}
}



// We do not really need init as all the layers should be already initialized
void MNeuralNet::Init(MyNeuralNet* net)
{
	Layers* layers = net->layers;
		
	layers->convLayer = new ConvLayer(4,4,2,32,32,new double[1]);
	layers->poolLayer = new PoolLayer(layers->convLayer);
	layers->FCLayer = new FCLayer(10000,10, layers->poolLayer);

}

void MNeuralNet::Evaluate(MyNeuralNet * net, string path)
{
}

void MNeuralNet::EvaluateOneFile(MyNeuralNet * net, string filePath)
{
	Layers* layers = net->layers;
	
	readFile(filePath, net->input);
	convertFormat(net);
	
	layers->convLayer->update_input(net->inputForNet);
	
	layers->convLayer->forward_layer;
	layers->poolLayer->forward_layer;
	layers->FCLayer->forward_layer;
}

void MNeuralNet::Learn(MyNeuralNet* net, string path)
{
	while (true) {
		LearnOneFile(net, path);
	}

}

void MNeuralNet::LearnOneFile(MyNeuralNet* net, std::string path)
{
	Layers* layers = net->layers;
	
	layers->convLayer->forward_layer();
	layers->poolLayer->forward_layer();
	layers->FCLayer->forward_layer();

	layers->FCLayer->backProp_layer();
	layers->poolLayer->backProp_layer();
	layers->convLayer->backProp_layer();

	layers->convLayer->learn();
	layers->poolLayer->learn();
	layers->FCLayer->learn();
}

void MNeuralNet::Release()
{

}

