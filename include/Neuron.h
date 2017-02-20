#ifndef _NEURON_H_
#define _NEURON_H_

#include <vector>
#include "Function.h"
#include "NeuralLink.h"

const double LearningRate = 0.001;
class NeuralLink;

class Neuron {
protected:
	std::vector<NeuralLink*> inputs;
	std::vector<NeuralLink*> outputs;
	Function* function;
	double totalSum;
	double learningRate;

public:
	Neuron();
	Neuron(const Neuron* _neuron);
	Neuron(Function *_function);
	Neuron(std::vector<NeuralLink*>& _outputs, Function* _function);
	Neuron(std::vector<Neuron *>& _outputs, Function* _function);
	virtual ~Neuron();

	virtual std::vector<NeuralLink*>& GetInputLinks();
	virtual std::vector<NeuralLink*>& GetOutputLinks();
	virtual NeuralLink* at(const int& indexOfNeuralLink);

	virtual void SetInputLink(NeuralLink* newNeuralLink);
	virtual void SetOutputLink(NeuralLink* newNeuralLink);

	virtual void Input(double inputData);
	virtual double Activation();
	virtual int GetNumOfOutputLinks();
	virtual double GetTotalSum();
	virtual void ResetTotalSum();
	virtual double Process();
	virtual double Process(double x);
	virtual double Derivative();

	virtual double TrainNeuron(double target);
	virtual void WeightsUpdate();
	virtual double CalculateLearningRate(double target);
	virtual void ShakeWeights();
	virtual void GetStatus();
	virtual double GetLearningRate();
};


class OutputNeuron : public Neuron
{
protected:
	double outputSum;

public:
	OutputNeuron();
	OutputNeuron(Neuron* _neuron);
	OutputNeuron(Function *_function);
	OutputNeuron(std::vector<NeuralLink*>& _outputs, Function* _function);
	OutputNeuron(std::vector<Neuron *>& _outputs, Function* _function);

	virtual	~OutputNeuron() {};
	
	virtual double	Activation();
	virtual double	TrainNeuron(double target);
	virtual void	WeightsUpdate();
	virtual double	CalculateLearningRate(double target);
	virtual void	ShakeWeights();
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
	virtual double	CalculateLearningRate(double target);
	virtual void	ShakeWeights();
	virtual void	GetStatus()								{ neuron->GetStatus(); };
	virtual double	GetLearningRate()						{ return neuron->GetLearningRate(); }
};

#endif // !_NEURON_H_