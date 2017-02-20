#ifndef _NEURALLINK_H
#define _NEURALLINK_H

class Neuron;

class NeuralLink {
public:
	NeuralLink();
	NeuralLink(Neuron *inNeuron, double inWeigth = 0.0);

	void SetWeigth(double inWeigth);
	double GetWeigth() const;

	void SetNeuron(Neuron* in);
	Neuron* GetNeuron() const;

	void SetWeigthCorrection(double inWeigthCorrection);
	double GetWeigthCorrection() const;

	void UpdateWeigth();
	void ShakeWeight();

	void SetLastTranslatedSignal(double inLastTranslatedSignal);
	double GetLastTranslatedSignal();

	void SetErrorInFormationTerm(double inEITerm);
	double GetErrorInFormationTerm();

protected:
	double weigth;
	Neuron* neuronNext;
	double weigthDelta;
	double lastTranslatedSignal;
	double errorInformationTerm;
};

#endif // !_NEURALLINK_H
