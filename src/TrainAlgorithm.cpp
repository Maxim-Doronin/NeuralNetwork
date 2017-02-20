#include "TrainAlgorithm.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void Backpropagation::hardInitialiazation()
{
	srand((unsigned)time(0));

	float numOfInputs = (float)neuralNetwork->inputs;
	float numOfHiddens = (float)neuralNetwork->hidden;
	float degree = 1.0f / numOfInputs;
	float scaleFactor = 0.7f * (pow(numOfHiddens, degree));

	int layerSize = neuralNetwork->size();
	for (int layerIdx = 0; layerIdx < layerSize; layerIdx++) {
		int neuronSize = (int)neuralNetwork->GetLayer(layerIdx).size();
		for (int neuronIdx = 0; neuronIdx < neuronSize; neuronIdx++) {
			Neuron* currentNeuron = neuralNetwork->GetLayer(layerIdx).at(neuronIdx);
			int outputsSize = currentNeuron->GetNumOfOutputLinks();
			for (int linkIdx = 0; linkIdx < outputsSize; linkIdx++) {
				NeuralLink* currentNeuralLink = currentNeuron->at(linkIdx);
				float weight = -0.5f + (float)rand() / (float)RAND_MAX;
				currentNeuralLink->SetWeigth(weight);
			}
		}
	}

	for (int layerIdx = 0; layerIdx < layerSize - 1; layerIdx++) {
		double dSquaredNorm = 0;

		int neuronSize = (int)neuralNetwork->GetLayer(layerIdx).size();
		for (int neuronIdx = 0; neuronIdx < neuronSize; neuronIdx++) {
			Neuron* currentNeuron = neuralNetwork->GetLayer(layerIdx).at(neuronIdx);
			int outputsSize = currentNeuron->GetNumOfOutputLinks();
			for (int linkIdx = 0; linkIdx < outputsSize; linkIdx++) {
				NeuralLink* currentNeuralLink = currentNeuron->at(linkIdx);
				dSquaredNorm += pow(currentNeuralLink->GetWeigth(), 2.0);
			}
		}
		double dNorm = sqrt(dSquaredNorm);

		for (int neuronIdx = 0; neuronIdx < neuronSize; neuronIdx++) {
			Neuron* currentNeuron = neuralNetwork->GetLayer(0).at(neuronIdx);
			int outputsSize = currentNeuron->GetNumOfOutputLinks();
			for (int linkIdx = 0; linkIdx < outputsSize; linkIdx++) {
				NeuralLink* currentNeuralLink = currentNeuron->at(linkIdx);
				double weight = (scaleFactor * (currentNeuralLink->GetWeigth())) / dNorm;
				currentNeuralLink->SetWeigth(weight);
			}
		}
	}

	//Bias init
	for (int layerIdx = 0; layerIdx < layerSize - 1; layerIdx++) {
		Neuron* Bias = neuralNetwork->GetBiasLayer().at(layerIdx);
		int biasSize = Bias->GetNumOfOutputLinks();
		for (int linkIdx = 0; linkIdx < biasSize; linkIdx++) {
			NeuralLink* currentNeuralLink = Bias->at(linkIdx);
			float weight = -scaleFactor + (float)rand() / ((float)RAND_MAX / (scaleFactor + scaleFactor));
			currentNeuralLink->SetWeigth(weight);
		}
	}
}

void Backpropagation::simpleInitialization()
{
	srand((unsigned)time(0));

	int layerSize = neuralNetwork->size();
	for (int layerIdx = 0; layerIdx < layerSize; layerIdx++) {
		int neuronSize = (int)neuralNetwork->GetLayer(layerIdx).size();
		for (int neuronIdx = 0; neuronIdx < neuronSize; neuronIdx++) {
			Neuron* currentNeuron = neuralNetwork->GetLayer(layerIdx).at(neuronIdx);
			int outputSize = currentNeuron->GetNumOfOutputLinks();
			for (int linkIdx = 0; linkIdx < outputSize; linkIdx++) {
				NeuralLink* currentNeuralLink = currentNeuron->at(linkIdx);
				float weight = -0.5f + (float)rand() / ((float)RAND_MAX);
				currentNeuralLink->SetWeigth(weight);
			}
		}
	}
	for (int layerIdx = 0; layerIdx < layerSize - 1; layerIdx++) {
		Neuron* Bias = neuralNetwork->GetBiasLayer().at(layerIdx);
		for (int linkIdx = 0; linkIdx < Bias->GetNumOfOutputLinks(); linkIdx++) {
			NeuralLink* currentNeuralLink = Bias->at(linkIdx);
			float weight = -0.5f + (float)rand() / ((float)RAND_MAX);
			currentNeuralLink->SetWeigth(weight);
		}
	}
}

void Backpropagation::WeightsInitialization()
{
	this->hardInitialiazation();
}

Backpropagation::Backpropagation(NeuralNetwork* _neuralNetwork)
{
	neuralNetwork = _neuralNetwork;
}

double Backpropagation::Train(const std::vector<double>& data, 
	const std::vector<double>& target)
{
	double result = 0;
	if (data.size() != neuralNetwork->inputs || 
		target.size() != neuralNetwork->outputs) {
		std::cout << "Incoorect size of data. Expected: " 
			<< neuralNetwork->inputs << " elements" << std::endl;
		return -1;
	}

	else {
		//precedent is applied to the network input
		for (int i = 0; i < neuralNetwork->inputs; i++) {
			neuralNetwork->GetInputLayer().at(i)->Input(data[i]);
		}

		//sequential activation of neurons
		int layerSize = neuralNetwork->size();
		for (int layerIdx = 0; layerIdx < layerSize - 1; layerIdx++) {
			neuralNetwork->GetBiasLayer().at(layerIdx)->Input(1);
			int neuronSize = (int)neuralNetwork->GetLayer(layerIdx).size();
			for (int neuronIdx = 0; neuronIdx < neuronSize; neuronIdx++) {
				neuralNetwork->GetLayer(layerIdx).at(neuronIdx)->Activation();
			}

			neuralNetwork->GetBiasLayer().at(layerIdx)->Activation();
			Neuron* biasNeuron = neuralNetwork->GetBiasLayer().at(layerIdx);
			int outputsSize = biasNeuron->GetNumOfOutputLinks();
			for (int i = 0; i < outputsSize; i++) {
				biasNeuron->GetOutputLinks().at(i)->SetLastTranslatedSignal(1);
			}
		}

		//get network response
		std::vector<double> netResponseYk;
		for (int outputsIdx = 0; outputsIdx < neuralNetwork->outputs; outputsIdx++) {
			double Yk = neuralNetwork->GetOutputLayer().at(outputsIdx)->Activation();
			netResponseYk.push_back(Yk);
		}

		//start trainig output neurons. Calculate MSE.
		for (int outputsIdx = 0; outputsIdx < neuralNetwork->outputs; outputsIdx++) {
			Neuron* neuron = neuralNetwork->GetOutputLayer().at(outputsIdx);
			result = neuron->TrainNeuron(target[outputsIdx]);
			neuralNetwork->AddMSE(result);
		}

		//train other neurons
		for (int layerIdx = layerSize - 2; layerIdx > 0; layerIdx--) {
			int neuronSize = (int)neuralNetwork->GetLayer(layerIdx).size();
			for (int neuronIdx = 0; neuronIdx < neuronSize; neuronIdx++) {
				neuralNetwork->GetLayer(layerIdx).at(neuronIdx)->TrainNeuron(0);
			}
		}

		neuralNetwork->UpdateWeights();
		neuralNetwork->ResetSums();
		return result;
	}
}