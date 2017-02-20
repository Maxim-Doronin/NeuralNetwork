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
	virtual double Process(double x);
	virtual double Derivative(double x);
};

class Sigmoid : public Function {
public:
	Sigmoid() {};
	virtual ~Sigmoid() {};
	virtual double Process(double x);
	virtual double Derivative(double x);
};

class BipolarSigmoid : public Function {
public:
	BipolarSigmoid() {};
	virtual ~BipolarSigmoid() {};
	virtual double Process(double x);
	virtual double Derivative(double x);
};

#endif // !_FUNCTION_H_