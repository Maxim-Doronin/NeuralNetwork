#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include "NeuralFactory.h"
#include "Function.h"
#include "TrainAlgorithm.h"
#include <cstring>
#include <iostream>
#include <fstream>

typedef unsigned char uchar;
class TrainAlgorithm;

class NeuralNetwork
{
public:
	NeuralNetwork(const int& _inputs, const int& _outputs, 
		const int& _numOfHiddenLayers = 0, const int& _hidden = 0, 
		const char* type = "MultiLayerPerceptron");
	NeuralNetwork(const char* filename);
	~NeuralNetwork();

	bool Train(const std::vector<std::vector<double> >& data, 
		const std::vector<std::vector<double> >& target);
	uchar GetNetResponse(const std::vector<double>& data);
	void SetAlgorithm(TrainAlgorithm* _trainingAlgorithm);
	void SetNeuronFactory(NeuronFactory* neuronFactory);
	void ShowNetworkState();
	const double& GetMinMSE();
	void SetMinMSE(const double& _minMse);

	void SaveParameters(const char* fileName);

	friend class Backpropagation;

protected:
	std::vector<Neuron *>& GetLayer(const int& idx);
	unsigned int size();
	std::vector<Neuron*>& GetOutputLayer();
	std::vector<Neuron*>& GetInputLayer();
	std::vector<Neuron*>& GetBiasLayer();
	void UpdateWeights();
	void ShakeWeights();
	void ResetSums();
	void AddMSE(double localMSE);
	double GetMSE();
	void ResetMSE();
	
	NeuronFactory* neuralFactory;				
	TrainAlgorithm* trainingAlgoritm;			
	std::vector<std::vector<Neuron*> > layers;						
	std::vector<Neuron *> biasLayer;					
	int inputs, outputs, hidden;	
	double meanSquaredError;			
	double minMSE;						
};

#endif // !_NEURALNETWORK_H_