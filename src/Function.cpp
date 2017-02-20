#include "Function.h"

double Linear::Process(double x)
{
	return x;
}

double Linear::Derivative(double x)
{
	return 0;
}

double Sigmoid::Process(double x)
{
	return ((double)1 / (1 + exp(-x)));
}

double Sigmoid::Derivative(double x)
{
	return (this->Process(x)*(1 - this->Process(x)));
}

double BipolarSigmoid::Process(double x)
{
	return ((double)2 / (1 + exp(-x)) - 1);
}

double BipolarSigmoid::Derivative(double x)
{
	return (0.5 * (1 + this->Process(x)) * (1 - this->Process(x)));
}