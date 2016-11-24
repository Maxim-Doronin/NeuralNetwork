#pragma once

#include "Neuron.h"

class NeuronFactory
{
public:
	NeuronFactory() {};
	virtual				~NeuronFactory() {};
	virtual Neuron *	CreateInputNeuron(std::vector<Neuron*>& inNeuronsLinkTo, Function* inNetFunc) = 0;
	virtual Neuron *	CreateOutputNeuron(Function * inNetFunc) = 0;
	virtual Neuron *	CreateHiddenNeuron(std::vector<Neuron*>& inNeuronsLinkTo, Function* inNetFunc) = 0;

};

class PerceptronNeuronFactory : public NeuronFactory
{
public:
	PerceptronNeuronFactory() {};
	virtual				~PerceptronNeuronFactory() {};
	virtual Neuron *	CreateInputNeuron(std::vector<Neuron*>& inNeuronsLinkTo, Function* inNetFunc)	{ return new Neuron(inNeuronsLinkTo, inNetFunc); };
	virtual Neuron * 	CreateOutputNeuron(Function * inNetFunc)										{ return new OutputNeuron(new Neuron(inNetFunc)); };
	virtual Neuron * 	CreateHiddenNeuron(std::vector<Neuron*>& inNeuronsLinkTo, Function* inNetFunc)	{ return new HiddenNeuron(new Neuron(inNeuronsLinkTo, inNetFunc)); };
};

