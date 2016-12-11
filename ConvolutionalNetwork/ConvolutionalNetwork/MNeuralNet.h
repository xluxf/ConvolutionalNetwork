#pragma once
#include "Layer.h"
#include <string>

#define DIM 32
#define DIM_SQR 1024

struct MyNeuralNet {
	Layers* layers;
	Input* input;
	std::vector<int> errors;
};

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


namespace MNeuralNet {
	void Init(Layers* layers);

	void Evaluate(MyNeuralNet* net, std::string path);

	void EvaluateOneFile(MyNeuralNet* net, std::string filePath);

	void Learn(MyNeuralNet* net, std::string path);
	
	void LearnOneFile(MyNeuralNet* net, std::string filePath);

	void Release();
}

