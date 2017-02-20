#include "Neuron.h"
#include <iostream>
#include <cmath>

#pragma region neuron
Neuron::Neuron()
{
	function = new Linear;
	totalSum = 0.0;
	learningRate = LearningRate;
}

Neuron::Neuron(const Neuron* _neuron)
{
	function = _neuron->function;
	totalSum = 0.0;
	learningRate = _neuron->learningRate;

	copy(_neuron->inputs.begin(), _neuron->inputs.end(), inputs.begin());
	copy(_neuron->outputs.begin(), _neuron->outputs.end(), outputs.begin());
}

Neuron::Neuron(Function *_function)
{
	function = _function;
	totalSum = 0.0;
	learningRate = LearningRate;
}

Neuron::Neuron(std::vector<NeuralLink*>& _outputs, Function* _function)
{
	function = _function;
	outputs = _outputs;
	totalSum = 0.0;
	learningRate = LearningRate;
}

Neuron::Neuron(std::vector<Neuron *>& _outputs, Function* _function) {
	function = _function;
	totalSum = 0.0;
	learningRate = LearningRate;

	for (int i = 0; i < _outputs.size(); i++) {
		NeuralLink *newLink = new NeuralLink(_outputs[i], 0.0);
		outputs.push_back(newLink);
		_outputs[i]->SetInputLink(newLink);
	}
}

Neuron::~Neuron() {
	delete function;
	for (int i = 0; i < outputs.size(); i++) {
		delete outputs[i];
	}
}

std::vector<NeuralLink*>& Neuron::GetInputLinks() 
{ 
	return inputs; 
}

std::vector<NeuralLink*>& Neuron::GetOutputLinks() 
{ 
	return outputs; 
}

NeuralLink* Neuron::at(const int& indexOfNeuralLink) 
{ 
	return outputs[indexOfNeuralLink]; 
}

void Neuron::SetInputLink(NeuralLink* newNeuralLink) 
{ 
	inputs.push_back(newNeuralLink); 
}

void Neuron::SetOutputLink(NeuralLink* newNeuralLink) 
{ 
	outputs.push_back(newNeuralLink); 
}

void Neuron::Input(double inputData) 
{ 
	totalSum += inputData; 
}
int Neuron::GetNumOfOutputLinks() 
{ 
	return (int)outputs.size(); 
}

double Neuron::GetTotalSum() 
{ 
	return totalSum; 
}

void Neuron::ResetTotalSum() 
{ 
	totalSum = 0.0; 
}

double Neuron::Process() 
{ 
	return function->Process(totalSum); 
}

double Neuron::Process(double x) 
{ 
	return function->Process(x); 
}
double Neuron::Derivative() 
{ 
	return function->Derivative(totalSum); 
}

double Neuron::TrainNeuron(double target) 
{ 
	return 0; 
}

void Neuron::WeightsUpdate() 
{ };

double Neuron::CalculateLearningRate(double target) 
{ 
	return 0.007; 
}

void Neuron::ShakeWeights() 
{ };

double Neuron::GetLearningRate() 
{ 
	return learningRate; 
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
		std::cout << "      Weight: " << currentLink->GetWeigth() 
			<< "; Weight correction term: " << currentLink->GetWeigthCorrection();
		std::cout << std::endl;
	}
}

#pragma endregion

#pragma region outputNeuron
OutputNeuron::OutputNeuron() : Neuron()
{ }

OutputNeuron::OutputNeuron(Neuron* _neuron) : Neuron(_neuron) 
{ }

OutputNeuron::OutputNeuron(Function *_function) : Neuron(_function)
{ }

OutputNeuron::OutputNeuron(std::vector<NeuralLink*>& _outputs, Function* _function)
	: Neuron(_outputs, _function)
{ }

OutputNeuron::OutputNeuron(std::vector<Neuron *>& _outputs, Function* _function)
	: Neuron (_outputs, _function)
{ }

double OutputNeuron::Activation() {
	double output = this->Process();
	outputSum = output;
	return output;
}

double OutputNeuron::TrainNeuron(double target)
{
	double res;
	double error = (target - outputSum) * this->Derivative();
	learningRate = CalculateLearningRate(target);
	res = pow(target - outputSum, 2);

	for (int i = 0; i < (this->GetInputLinks()).size(); i++) {
		NeuralLink* inputLink = (this->GetInputLinks()).at(i);
		
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

#pragma endregion

#pragma region hiddenNeuron
HiddenNeuron::HiddenNeuron() : Neuron()
{ }

HiddenNeuron::HiddenNeuron(Neuron* _neuron) : Neuron(_neuron)
{ }

HiddenNeuron::HiddenNeuron(Function *_function) : Neuron(_function)
{ }

HiddenNeuron::HiddenNeuron(std::vector<NeuralLink*>& _outputs, Function* _function)
	: Neuron(_outputs, _function)
{ }

HiddenNeuron::HiddenNeuron(std::vector<Neuron *>& _outputs, Function* _function)
	: Neuron(_outputs, _function)
{ }

double HiddenNeuron::Activation()
{
	for (int i = 0; i < this->GetNumOfOutputLinks(); i++) {

		NeuralLink* currentLink = this->at(i);
		Neuron* currentNeuronLinkedTo = currentLink->GetNeuron();

		double weight = currentLink->GetWeigth();
		double sum = this->GetTotalSum();
		double zj = (this->Process(sum));
		double output = zj * weight;

		currentLink->SetLastTranslatedSignal(zj);
		currentNeuronLinkedTo->Input(output);
	}
	return this->GetTotalSum();
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

#pragma endregion