#pragma once
#include "NeuralFactory.h"
#include "Function.h"
#include "TrainAlgorithm.h"
#include <cstring>
#include <iostream>

typedef unsigned char uchar;
class TrainAlgorithm;

class NeuralNetwork
{
public:
	NeuralNetwork(const int& _inputs, const int& _outputs, const int& _numOfHiddenLayers = 0,
		const int& _hidden = 0, const char* type = "MultiLayerPerceptron");
	~NeuralNetwork();

	bool Train(const std::vector<std::vector<double> >& data, const std::vector<std::vector<double> >& target);
	uchar GetNetResponse(const std::vector<double>& data);
	void SetAlgorithm(TrainAlgorithm* _trainingAlgorithm)	{ trainingAlgoritm = _trainingAlgorithm; };
	void SetNeuronFactory(NeuronFactory* neuronFactory)		{ neuralFactory = neuronFactory; };
	void ShowNetworkState();
	const double& GetMinMSE()								{ return minMSE; };
	void SetMinMSE(const double& _minMse)					{ minMSE = _minMse; };

	friend class Backpropagation;

protected:
	std::vector<Neuron *>&	GetLayer(const int& idx)		{ return layers[idx]; };
	unsigned int size()										{ return layers.size(); };
	std::vector<Neuron*>&	GetOutputLayer()				{ return layers[layers.size() - 1]; };
	std::vector<Neuron*>&	GetInputLayer()					{ return layers[0]; };
	std::vector<Neuron*>& 	GetBiasLayer()					{ return biasLayer; };
	void UpdateWeights();
	void ResetSums();
	void AddMSE(double localMSE)							{ meanSquaredError += localMSE; };
	double	GetMSE()										{ return meanSquaredError; };
	void ResetMSE()											{ meanSquaredError = 0; };


	NeuronFactory*						neuralFactory;				
	TrainAlgorithm*						trainingAlgoritm;			
	std::vector<std::vector<Neuron*> > 	layers;						
	std::vector<Neuron *> 				biasLayer;					
	int									inputs, outputs, hidden;	
	double								meanSquaredError;			
	double								minMSE;						
};
