#pragma once
#include "Layer.h"
#include <vector>
#include <string>

#define DIM 32
#define DIM_SQR 1024

struct Layers {
	Layer* FCLayer;
	Layer* convLayer;
	Layer* poolLayer;
};

struct Input {
	char label;
	char red[DIM_SQR];
	char green[DIM_SQR];
	char blue[DIM_SQR];
};

struct MyNeuralNet {
	Layers* layers;
	Input* input;
	double* inputForNet;
	std::vector<int> errors;
};

namespace MNeuralNet {
	void Init(MyNeuralNet* net);

	void Evaluate(MyNeuralNet* net, std::string path);

	void EvaluateOneFile(MyNeuralNet* net, std::string filePath);

	void Learn(MyNeuralNet* net, std::string path);
	
	void LearnOneFile(MyNeuralNet* net, std::string filePath);

	void Release();
}

