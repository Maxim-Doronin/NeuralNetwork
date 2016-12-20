#include "Neuron.h"
#include <iostream>
#include <cmath>

Neuron::Neuron(std::vector<Neuron *>& neuronsLinkTo, Function* _function) {
	function = _function;
	totalSum = 0.0;
	learningRate = LearningRate;

	for (int i = 0; i < neuronsLinkTo.size(); i++) {
		NeuralLink *newLink = new NeuralLink(neuronsLinkTo[i], 0.0);
		outputs.push_back(newLink);
		neuronsLinkTo[i]->SetInputLink(newLink);
	}
}

Neuron::~Neuron() {
	delete function;
	for (int i = 0; i < outputs.size(); i++) {
		delete outputs[i];
	}
}

double	Neuron::Activation() {
	for (int i = 0; i < this->GetNumOfOutputLinks(); i++) {
		NeuralLink* currentLink = outputs[i];
		Neuron* currentNeuronLinkedTo = currentLink->GetNeuron();

		double weight = currentLink->GetWeigth();
		double sum = totalSum;
		double xi = (function->Process(sum));
		double output = xi*weight;

		currentLink->SetLastTranslatedSignal(xi);
		currentNeuronLinkedTo->Input(output);
	}
	return totalSum;
}

void Neuron::GetStatus() {
	for (int i = 0; i < outputs.size(); i++) {
		NeuralLink* currentLink = outputs.at(i);
		std::cout << "    Link index: " << i << std::endl;
		std::cout << "      Weight: " << currentLink->GetWeigth() << "; Weight correction term: " << currentLink->GetWeigthCorrection();
		std::cout << std::endl;
	}
}


double OutputNeuron::Activation() {
	double output = this->Process();
	outputSum = output;
	return output;
}

double OutputNeuron::TrainNeuron(double target)
{
	double res;
	double error = (target - outputSum) * neuron->Derivative();
	learningRate = CalculateLearningRate(target);
	res = pow(target - outputSum, 2);

	for (int i = 0; i < (this->GetInputLinks()).size(); i++) {
		NeuralLink * inputLink = (this->GetInputLinks()).at(i);
		
		double Zj = inputLink->GetLastTranslatedSignal();
		double weightCorrectionTerm = Zj * error;
		
		inputLink->SetWeigthCorrection(LearningRate * weightCorrectionTerm);
		inputLink->SetErrorInFormationTerm(error);
	}

	return res;
}

double OutputNeuron::CalculateLearningRate(double target) {
	double a = 0.0;
	double b = 1000.0;
	double eps = 0.0001;
	double delta = 0.0001;

	while ((b - a) / 2 >= eps) {
		double lambda1 = (a + b - delta) / 2;
		double lambda2 = (a + b + delta) / 2;
		if (pow((target - Process(totalSum - lambda1 * Derivative())), 2) >
			pow((target - Process(totalSum - lambda2 * Derivative())), 2))
		//if (Process(totalSum - lambda1 * Derivative()) > Process(totalSum - lambda2 * Derivative()))
			a = lambda1;
		else
			b = lambda2;
	}
	return (a + b) / 2;
}

void OutputNeuron::WeightsUpdate()
{
	for (int i = 0; i < (this->GetInputLinks()).size(); i++) {
		NeuralLink* inputLink = (this->GetInputLinks()).at(i);
		inputLink->UpdateWeigth();
	}
}

void OutputNeuron::ShakeWeights() 
{
	for (int i = 0; i < (this->GetInputLinks()).size(); i++) {
		NeuralLink* inputLink = (this->GetInputLinks()).at(i);
		inputLink->ShakeWeight();
	}
}





double HiddenNeuron::Activation()
{
	for (int i = 0; i < this->GetNumOfOutputLinks(); i++) {

		NeuralLink* currentLink = neuron->at(i);
		Neuron* currentNeuronLinkedTo = currentLink->GetNeuron();

		double weight = currentLink->GetWeigth();
		double sum = neuron->GetTotalSum();
		double zj = (neuron->Process(sum));
		double output = zj * weight;

		currentLink->SetLastTranslatedSignal(zj);
		currentNeuronLinkedTo->Input(output);
	}
	return neuron->GetTotalSum();
}

double HiddenNeuron::TrainNeuron(double target)
{
	double deltaInputs = 0;
	for (int i = 0; i < (this->GetNumOfOutputLinks()); i++) {
		NeuralLink* outputLink = (this->GetOutputLinks()).at(i);
		double error = outputLink->GetErrorInFormationTerm();
		double weight = outputLink->GetWeigth();
		deltaInputs += (weight * error);
	}

	double errorj = deltaInputs * (this->Derivative());
	learningRate = CalculateLearningRate(target);

	for (int i = 0; i < (this->GetInputLinks()).size(); i++) {
		NeuralLink* inputLink = (this->GetInputLinks()).at(i);
		
		double Xi = inputLink->GetLastTranslatedSignal();
		double weightCorrectionTerm = Xi * errorj;
		
		inputLink->SetWeigthCorrection(LearningRate * weightCorrectionTerm);
		inputLink->SetErrorInFormationTerm(errorj);
	}
	return 0;
}

double HiddenNeuron::CalculateLearningRate(double target) {
	double a = 0.0;
	double b = 1000.0;
	double eps = 0.0001;
	double delta = 0.0001;

	while ((b - a) / 2 >= eps) {
		double lambda1 = (a + b - delta) / 2;
		double lambda2 = (a + b + delta) / 2;
		if (pow((target - Process(totalSum - lambda1 * Derivative())), 2) >
			pow((target - Process(totalSum - lambda2 * Derivative())), 2))
			//if (Process(totalSum - lambda1 * Derivative()) > Process(totalSum - lambda2 * Derivative()))
			a = lambda1;
		else
			b = lambda2;
	}
	return (a + b) / 2;
}

void HiddenNeuron::WeightsUpdate()
{
	for (int i = 0; i < (this->GetInputLinks()).size(); i++) {
		NeuralLink* inputLink = (this->GetInputLinks()).at(i);
		inputLink->UpdateWeigth();
	}
}

void HiddenNeuron::ShakeWeights()
{
	for (int i = 0; i < (this->GetInputLinks()).size(); i++) {
		NeuralLink* inputLink = (this->GetInputLinks()).at(i);
		inputLink->ShakeWeight();
	}
}