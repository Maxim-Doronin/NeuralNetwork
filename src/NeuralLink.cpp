#include "NeuralLink.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

NeuralLink::NeuralLink() 
{
	weigth = 0.0;
	neuronNext = 0;
	weigthDelta = 0.0;
}

NeuralLink::NeuralLink(Neuron *inNeuron, double inWeigth) 
{
	weigth = inWeigth;
	neuronNext = inNeuron;
	weigthDelta = 0.0;
}

void NeuralLink::SetWeigth(const double inWeigth) 
{
	weigth = inWeigth;
}

double NeuralLink::GetWeigth() const 
{
	return weigth;
}

void NeuralLink::SetNeuron(Neuron* in) 
{
	neuronNext = in;
}

Neuron* NeuralLink::GetNeuron() const
{
	return neuronNext;
}

void NeuralLink::SetWeigthCorrection(const double inWeigthCorrection) 
{
	weigthDelta = inWeigthCorrection;
}

double NeuralLink::GetWeigthCorrection() const 
{
	return weigthDelta;
}

void NeuralLink::UpdateWeigth() 
{
	weigth += weigthDelta;
}

void NeuralLink::ShakeWeight() 
{
	srand((unsigned)time(0));
	double offset = ((float)rand() / RAND_MAX) / 2 * weigth;
	weigth += offset - weigth / 4;
}

void NeuralLink::SetLastTranslatedSignal(double inLastTranslatedSignal) 
{
	lastTranslatedSignal = inLastTranslatedSignal;
};

double NeuralLink::GetLastTranslatedSignal() 
{
	return lastTranslatedSignal;
}

void NeuralLink::SetErrorInFormationTerm(double inEITerm) 
{
	errorInformationTerm = inEITerm;
}

double NeuralLink::GetErrorInFormationTerm() 
{
	return errorInformationTerm;
}

