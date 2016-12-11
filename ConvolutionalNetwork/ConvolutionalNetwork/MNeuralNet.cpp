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


// We do not really need init as all the layers should be already initialized
void MNeuralNet::Init(Layers* layers)
{
	/*layers->FCLayer = new FCLayer(, NUMBER_OF_NEURONS);
	layers->convLayer = new ConvLayer();
	layers->poolLayer = new PoolLayer;*/
}

void MNeuralNet::Evaluate(MyNeuralNet * net, string path)
{
}

void MNeuralNet::EvaluateOneFile(MyNeuralNet * net, string filePath)
{
	readFile(filePath, net->input);
	Layers* layers = net->layers;
	
	//layers->convLayer->setInput;
	layers->convLayer->forward_layer;
	//layers->poolLayer->setInput;
	layers->poolLayer->forward_layer;
	//layers->FClayer->setInput
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

	/*layers->FCLayer->backProp_layer();
	layers->poolLayer->backProp_layer();
	layers->convLayer->backProp_layer();*/
}

void MNeuralNet::Release()
{

}

