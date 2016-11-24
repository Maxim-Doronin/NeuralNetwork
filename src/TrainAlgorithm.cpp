#include "TrainAlgorithm.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void Backpropagation::hardInitialiazation()
{
	srand((unsigned)time(0));

	double numOfInputs = neuralNetwork->inputs;
	double numOfHiddens = neuralNetwork->hidden;
	double degree = 1.0 / numOfInputs;
	double scaleFactor = 0.7*(pow(numOfHiddens, degree));

	for (int layerIdx = 0; layerIdx < neuralNetwork->size(); layerIdx++) {
		for (int neuronIdx = 0; neuronIdx < neuralNetwork->GetLayer(layerIdx).size(); neuronIdx++) {
			Neuron* currentNeuron = neuralNetwork->GetLayer(layerIdx).at(neuronIdx);

			for (int linkIdx = 0; linkIdx < currentNeuron->GetNumOfOutputLinks(); linkIdx++) {
				NeuralLink* currentNeuralLink = currentNeuron->at(linkIdx);
				float pseudoRandWeight = -0.5 + (float)rand() / (float)RAND_MAX;
				currentNeuralLink->SetWeigth(pseudoRandWeight);
			}
		}
	}

	for (int neuronHiddenIdx = 0; neuronHiddenIdx < neuralNetwork->GetLayer(1).size(); neuronHiddenIdx++) {
		double dSquaredNorm = 0;

		for (int neuronInputIdx = 0; neuronInputIdx < neuralNetwork->GetLayer(0).size(); neuronInputIdx++) {
			Neuron* currentInputNeuron = neuralNetwork->GetLayer(0).at(neuronInputIdx);
			NeuralLink* currentNeuralLink = currentInputNeuron->at(neuronHiddenIdx);
			dSquaredNorm += pow(currentNeuralLink->GetWeigth(), 2.0);
		}

		double dNorm = sqrt(dSquaredNorm);

		for (int neuronInputIdx = 0; neuronInputIdx < neuralNetwork->GetLayer(0).size(); neuronInputIdx++) {
			Neuron* currentInputNeuron = neuralNetwork->GetLayer(0).at(neuronInputIdx);
			NeuralLink* currentNeuralLink = currentInputNeuron->at(neuronHiddenIdx);
			double newWeight = (scaleFactor * (currentNeuralLink->GetWeigth())) / dNorm;
			currentNeuralLink->SetWeigth(newWeight);
		}

	}

	//Bias init
	for (int layerIdx = 0; layerIdx < neuralNetwork->size() - 1; layerIdx++) {
		Neuron* Bias = neuralNetwork->GetBiasLayer().at(layerIdx);
		for (int linkIdx = 0; linkIdx < Bias->GetNumOfOutputLinks(); linkIdx++) {
			NeuralLink* currentNeuralLink = Bias->at(linkIdx);
			float pseudoRandWeight = -scaleFactor + (float)rand() / ((float)RAND_MAX / (scaleFactor + scaleFactor));
			currentNeuralLink->SetWeigth(pseudoRandWeight);
		}
	}
}

void Backpropagation::simpleInitialization()
{
	srand((unsigned)time(0));

	for (unsigned int layerInd = 0; layerInd < neuralNetwork->size(); layerInd++) {
		for (unsigned int neuronInd = 0; neuronInd < neuralNetwork->GetLayer(layerInd).size(); neuronInd++) {
			Neuron* currentNeuron = neuralNetwork->GetLayer(layerInd).at(neuronInd);
			for (int linkInd = 0; linkInd < currentNeuron->GetNumOfOutputLinks(); linkInd++) {
				NeuralLink* currentNeuralLink = currentNeuron->at(linkInd);
				float pseudoRandWeight = -0.5 + (float)rand() / ((float)RAND_MAX / (0.5 + 0.5));
				currentNeuralLink->SetWeigth(pseudoRandWeight);
			}
		}
	}
	for (unsigned int layerInd = 0; layerInd < neuralNetwork->size() - 1; layerInd++) {
		Neuron* Bias = neuralNetwork->GetBiasLayer().at(layerInd);
		for (int linkInd = 0; linkInd < Bias->GetNumOfOutputLinks(); linkInd++) {
			NeuralLink* currentNeuralLink = Bias->at(linkInd);
			float pseudoRandWeight = -0.5 + (float)rand() / ((float)RAND_MAX / (0.5 + 0.5));
			currentNeuralLink->SetWeigth(pseudoRandWeight);
		}
	}
}

void Backpropagation::WeightsInitialization()
{
	this->hardInitialiazation();
}

Backpropagation::Backpropagation(NeuralNetwork* _neuralNetwork)
{
	neuralNetwork = neuralNetwork;
}

double Backpropagation::Train(const std::vector<double>& data, const std::vector<double>& target)
{
	double result = 0;
	if (data.size() != neuralNetwork->inputs || target.size() != neuralNetwork->outputs) {
		std::cout << "Incoorect size of data. Expected: " << neuralNetwork->inputs << " elements" << std::endl;
		return -1;
	}

	else {
		//precedent is applied to the network input
		for (int i = 0; i < neuralNetwork->inputs; i++) {
			neuralNetwork->GetInputLayer().at(i)->Input(data[i]);
		}

		//sequential activation of neurons
		for (int ilayer = 0; ilayer < neuralNetwork->size() - 1; ilayer++) {
			neuralNetwork->GetBiasLayer().at(ilayer)->Input(1);
			for (int ineuron = 0; ineuron < neuralNetwork->GetLayer(ilayer).size(); ineuron++) {
				neuralNetwork->GetLayer(ilayer).at(ineuron)->Activation();
			}

			neuralNetwork->GetBiasLayer().at(ilayer)->Activation();
			for (int i = 0; i < neuralNetwork->GetBiasLayer().at(ilayer)->GetNumOfOutputLinks(); i++) {
				neuralNetwork->GetBiasLayer().at(ilayer)->GetOutputLinks().at(i)->SetLastTranslatedSignal(1);
			}
		}

		//get network response
		std::vector<double> netResponseYk;
		for (int ioutputs = 0; ioutputs < neuralNetwork->outputs; ioutputs++) {
			double Yk = neuralNetwork->GetOutputLayer().at(ioutputs)->Activation();
			netResponseYk.push_back(Yk);
		}

		//start trainig output neurons. Calculate MSE.
		for (int ioutputs = 0; ioutputs < neuralNetwork->outputs; ioutputs++) {
			result = neuralNetwork->GetOutputLayer().at(ioutputs)->TrainNeuron(target[ioutputs]);
			neuralNetwork->AddMSE(result);
		}

		//train other neurons
		for (int ilayer = neuralNetwork->size() - 2; ilayer > 0; ilayer--) {
			for (int ineuron = 0; ineuron < neuralNetwork->GetLayer(ilayer).size(); ineuron++) {
				neuralNetwork->GetLayer(ilayer).at(ineuron)->TrainNeuron(0);
			}
		}

		neuralNetwork->UpdateWeights();
		neuralNetwork->ResetSums();
		return result;
	}
}