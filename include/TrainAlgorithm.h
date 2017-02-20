#ifndef _TRAINALGORITHM_H_
#define _TRAINALGORITHM_H_

#include <vector>
#include "NeuralNetwork.h"

typedef unsigned char uchar;
class NeuralNetwork;

class TrainAlgorithm
{
public:
	virtual ~TrainAlgorithm() {};
	virtual double Train(const std::vector<double>& data, 
		const std::vector<double>& target) = 0;
	virtual void WeightsInitialization() = 0;
};

class Backpropagation : public TrainAlgorithm
{
public:
	Backpropagation(NeuralNetwork * _neuralNetwork);
	virtual	~Backpropagation() {};

	virtual double Train(const std::vector<double>& data, 
		const std::vector<double>& target);
	virtual void WeightsInitialization();

protected:
	void hardInitialiazation();
	void simpleInitialization();
	NeuralNetwork* neuralNetwork;
};

#endif // !_TRAINALGORITHM_H_