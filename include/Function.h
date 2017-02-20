#ifndef _FUNCTION_H_
#define _FUNCTION_H_

#include <cmath>

class Function {
public:
	Function() {};
	virtual ~Function() {};
	virtual double Process(double x) = 0;
	virtual double Derivative(double x) = 0;
};

class Linear : public Function {
public:
	Linear() {};
	virtual ~Linear() {};
	virtual double Process(double x) { return x; };
	virtual double Derivative(double x) { return 0; };
};

class Sigmoid : public Function {
public:
	Sigmoid() {};
	virtual ~Sigmoid() {};
	virtual double 	Process(double x) { return ((double)1 / (1 + exp(-x))); };
	virtual double 	Derivative(double x) { return (this->Process(x)*(1 - this->Process(x))); };
};

class BipolarSigmoid : public Function {
public:
	BipolarSigmoid() {};
	virtual ~BipolarSigmoid() {};
	virtual double 	Process(double x) { return ((double)2 / (1 + exp(-x)) - 1); };
	virtual double 	Derivative(double x) { return (0.5 * (1 + this->Process(x)) * (1 - this->Process(x))); };
};

#endif // !_FUNCTION_H_