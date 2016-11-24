#include "Neuron.h"
#include <iostream>

Neuron::Neuron(std::vector<Neuron *>& neuronsLinkTo, Function* _function) {
	function = _function;
	totalSum = 0.0;

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
	double error = (target - outputSum) *neuron->Derivative();
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

void OutputNeuron::WeightsUpdate()
{
	for (int i = 0; i < (this->GetInputLinks()).size(); i++) {
		NeuralLink* inputLink = (this->GetInputLinks()).at(i);
		inputLink->UpdateWeigth();
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

	for (int i = 0; i < (this->GetInputLinks()).size(); i++) {
		NeuralLink* inputLink = (this->GetInputLinks()).at(i);
		
		double Xi = inputLink->GetLastTranslatedSignal();
		double weightCorrectionTerm = Xi * errorj;
		
		inputLink->SetWeigthCorrection(LearningRate * weightCorrectionTerm);
		inputLink->SetErrorInFormationTerm(errorj);
	}
	return 0;
}

void HiddenNeuron::WeightsUpdate()
{
	for (int i = 0; i < (this->GetInputLinks()).size(); i++) {
		NeuralLink* inputLink = (this->GetInputLinks()).at(i);
		inputLink->UpdateWeigth();
	}
}