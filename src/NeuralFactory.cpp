#include "NeuralFactory.h"

Neuron*	PerceptronNeuronFactory::CreateInputNeuron
	(std::vector<Neuron*>& inNeuronsLinkTo,	Function* inNetFunc)
{
	return new Neuron(inNeuronsLinkTo, inNetFunc);
}

Neuron* PerceptronNeuronFactory::CreateHiddenNeuron
	(std::vector<Neuron*>& inNeuronsLinkTo,	Function* inNetFunc)
{
	return new HiddenNeuron(new Neuron(inNeuronsLinkTo, inNetFunc));
}

Neuron* PerceptronNeuronFactory::CreateOutputNeuron(Function * inNetFunc)
{
	return new OutputNeuron(new Neuron(inNetFunc));
}