#pragma once
#include <vector>
#include "Function.h"
#include "NeuralLink.h"

const double LearningRate = 0.007;
class NeuralLink;

class Neuron {
protected:
	std::vector<NeuralLink*> inputs;
	std::vector<NeuralLink*> outputs;
	Function* function;
	double totalSum;

public:
	Neuron() : function(new Linear), totalSum(0.0) {};
	Neuron(Function *_function) : function(_function), totalSum(0.0) {};
	Neuron(std::vector<NeuralLink*>& _outputLinks, Function* _function) : function(_function), outputs(_outputLinks), totalSum(0.0) {};
	Neuron(std::vector<Neuron *>& neuronsLinkTo, Function* _function);
	virtual ~Neuron();

	virtual std::vector<NeuralLink*>&	GetInputLinks()		{ return inputs; }
	virtual std::vector<NeuralLink*>&	GetOutputLinks()	{ return outputs; }
	virtual NeuralLink*	at(const int& indexOfNeuralLink)	{ return outputs[indexOfNeuralLink]; }

	virtual void SetInputLink(NeuralLink* newNeuralLink)	{ inputs.push_back(newNeuralLink); }
	virtual void SetOutputLink(NeuralLink* newNeuralLink)	{ outputs.push_back(newNeuralLink); }

	virtual void	Input(double inputData)					{ totalSum += inputData; };
	virtual double	Activation();
	virtual int		GetNumOfOutputLinks()					{ return outputs.size(); }
	virtual double	GetTotalSum()							{ return totalSum; }
	virtual void	ResetTotalSum()							{ totalSum = 0.0; }
	virtual double	Process()								{ return function->Process(totalSum); }
	virtual double	Process(double x)						{ return function->Process(x); }
	virtual double	Derivative()							{ return function->Derivative(totalSum); }

	virtual double	TrainNeuron(double target)				{ return 0; }
	virtual void	WeightsUpdate()							{ };
	virtual void	GetStatus();
};


class OutputNeuron : public Neuron
{
protected:
	double outputSum;
	Neuron*	neuron;

public:
	OutputNeuron(Neuron* inNeuron) { outputSum = 0; neuron = inNeuron; };
	virtual	~OutputNeuron() { delete neuron; }

	virtual std::vector<NeuralLink*>& GetInputLinks()		{ return neuron->GetInputLinks(); };
	virtual std::vector<NeuralLink*>& GetOutputLinks()		{ return neuron->GetOutputLinks(); };
	virtual NeuralLink* at(const int& inIndexOfNeuralLink)	{ return (neuron->at(inIndexOfNeuralLink)); };

	virtual void SetInputLink(NeuralLink* newNeuralLink)	{ neuron->SetInputLink(newNeuralLink); };
	virtual void SetOutputLink(NeuralLink* newNeuralLink)	{ neuron->SetOutputLink(newNeuralLink); };

	virtual void	Input(double inputData)					{ neuron->Input(inputData); };
	virtual double	Activation();
	virtual int		GetNumOfOutputLinks()					{ return neuron->GetNumOfOutputLinks(); };
	virtual double	GetTotalSum()							{ return neuron->GetTotalSum(); };
	virtual void	ResetTotalSum()							{ neuron->ResetTotalSum(); };
	virtual double	Process()								{ return neuron->Process(); };
	virtual double	Process(double x)						{ return neuron->Process(x); };
	virtual double	Derivative()							{ return neuron->Derivative(); };

	virtual double	TrainNeuron(double target);
	virtual void	WeightsUpdate();
	virtual void	GetStatus()								{ neuron->GetStatus(); };
};



class HiddenNeuron : public Neuron
{
protected:
	Neuron*	neuron;

public:
	HiddenNeuron(Neuron* inNeuron)	{ neuron = inNeuron; };
	virtual	~HiddenNeuron()			{ delete neuron; }

	virtual std::vector<NeuralLink*>&	GetInputLinks()		{ return neuron->GetInputLinks(); };
	virtual std::vector<NeuralLink*>&	GetOutputLinks()	{ return neuron->GetOutputLinks(); };
	virtual NeuralLink*	 at(const int& inIndexOfNeuralLink) { return (neuron->at(inIndexOfNeuralLink)); };

	virtual void	SetInputLink(NeuralLink* newNeuralLink) { neuron->SetInputLink(newNeuralLink); };
	virtual void	SetOutputLink(NeuralLink* newNeuralLink){ neuron->SetOutputLink(newNeuralLink); };

	virtual void	Input(double inputData)					{ neuron->Input(inputData); };
	virtual double	Activation();
	virtual int		GetNumOfOutputLinks()					{ return neuron->GetNumOfOutputLinks(); };
	virtual double	GetTotalSum()							{ return neuron->GetTotalSum(); };
	virtual void	ResetTotalSum()							{ neuron->ResetTotalSum(); };
	virtual double	Process()								{ return neuron->Process(); };
	virtual double	Process(double x)						{ return neuron->Process(x); };
	virtual double	Derivative()							{ return neuron->Derivative(); };

	virtual double	TrainNeuron(double target);
	virtual void	WeightsUpdate();
	virtual void	GetStatus()								{ neuron->GetStatus(); };
};
