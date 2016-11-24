#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const int& _inputs, const int& _outputs, const int& _numOfHiddenLayers = 0,
			const int& _hidden = 0, const char* type = "MultiLayerPerceptron")
{
	if (_inputs < 0 || _outputs < 0)
		std::cout << "Error in Neural Network constructor: The number of input and output neurons has to be more than 0!\n"; 
	
	else {
		minMSE = 0.01;
		meanSquaredError = 0;
		inputs = _inputs;
		outputs = _outputs;
		hidden = _hidden;

		Function* OutputNeuronsFunc;
		Function* InputNeuronsFunc;

		std::vector<Neuron*> outputLayer;
		std::vector<Neuron*> inputLayer;

		if (strcmp(type, "MultiLayerPerceptron") == 0) {
			neuralFactory = new PerceptronNeuronFactory;
			trainingAlgoritm = new Backpropagation(this);

			OutputNeuronsFunc = new Sigmoid;
			InputNeuronsFunc = new Linear;

		}

		//create output neurons
		for (int i = 0; i < outputs; i++) {
			outputLayer.push_back(neuralFactory->CreateOutputNeuron(OutputNeuronsFunc));
		}
		layers.push_back(outputLayer);

		//create hidden neurons
		for (int i = 0; i < _numOfHiddenLayers; i++) {
			std::vector<Neuron*> HiddenLayer;
			for (int j = 0; j < hidden; j++) {
				Neuron* hidden = neuralFactory->CreateHiddenNeuron(layers[0], OutputNeuronsFunc);
				HiddenLayer.push_back(hidden);
			}
			biasLayer.insert(biasLayer.begin(), neuralFactory->CreateInputNeuron(layers[0], InputNeuronsFunc));
			layers.insert(layers.begin(), HiddenLayer);
		}

		//create input neurons
		for (int i = 0; i < inputs; i++) {
			inputLayer.push_back(neuralFactory->CreateInputNeuron(layers[0], InputNeuronsFunc));
		}
		biasLayer.insert(biasLayer.begin(), neuralFactory->CreateInputNeuron(layers[0], InputNeuronsFunc));
		layers.insert(layers.begin(), inputLayer);

		trainingAlgoritm->WeightsInitialization();
	}
}

NeuralNetwork::~NeuralNetwork()
{
	delete neuralFactory;
	delete trainingAlgoritm;

	for (int i = 0; i < biasLayer.size(); i++) {
		delete biasLayer[i];
	}

	for (int i = 0; i < layers.size(); i++) {
		for (int j = 0; j < layers[i].size(); j++) {
			delete layers[i].at(j);
		}
	}

}

bool NeuralNetwork::Train(const std::vector<std::vector<double> >& data, const std::vector<std::vector<double> >& target)
{
	bool flag = true;
	int iterations = 0;
	while (flag && iterations < 3) {
		iterations++;
		std::cout << "--Start algorithm" << std::endl;
		for (int i = 0; i < data.size(); i++) {
			trainingAlgoritm->Train(data[i], target[i]);
		}
		
		double MSE = this->GetMSE();
		std::cout << "At " << iterations << " iteration MSE: " << MSE << " was achieved\n";
		if (MSE < minMSE) {
			std::cout << "At " << iterations << " iteration MSE: " << MSE << " was achieved\n";
			flag = false;
		}
		this->ResetMSE();
	}
	return flag;
}

uchar NeuralNetwork::GetNetResponse(const std::vector<double>& inData)
{
	std::vector<int> netResponse;
	if (inData.size() != inputs) {
		std::cout << "Input data dimensions are wrong, expected: " << inputs << " elements\n";
		return 11;
	}
	else {
		for (int i = 0; i < this->GetInputLayer().size(); i++) {
			this->GetInputLayer().at(i)->Input(inData[i]);
		}

		for (int numOfLayers = 0; numOfLayers < layers.size() - 1; numOfLayers++) {
			biasLayer[numOfLayers]->Input(1);

			for (int indexOfData = 0; indexOfData < layers.at(numOfLayers).size(); indexOfData++) {
				layers.at(numOfLayers).at(indexOfData)->Activation();
			}

			biasLayer[numOfLayers]->Activation();
		}
		double maxRes = -1;
		int answer;
		std::cout << "Net response is: { ";
		for (int ioutput = 0; ioutput < outputs; ioutput++) {
			double res = this->GetOutputLayer().at(ioutput)->Activation();
			if (res > maxRes) {
				maxRes = res;
				answer = ioutput;
			}
			std::cout.precision(4);
			std::cout.setf(std::ios::fixed);
			std::cout << res << "\t";
			netResponse.push_back(res);
		}
		std::cout << "|| " << answer << " } ";
		this->ResetSums();
		return answer;
	}
}

void NeuralNetwork::ResetSums()
{
	for (int i = 0; i < layers.size(); i++)
		for (unsigned int indexOfOutputElements = 0; indexOfOutputElements < layers.at(i).size(); indexOfOutputElements++)
			layers.at(i).at(indexOfOutputElements)->ResetTotalSum();

	for (unsigned int i = 0; i < layers.size() - 1; i++)
		biasLayer[i]->ResetTotalSum();
}

void NeuralNetwork::UpdateWeights()
{
	for (int idxOfLayer = 0; idxOfLayer < layers.size(); idxOfLayer++) {
		for (int idxOfNeuron = 0; idxOfNeuron < layers[idxOfLayer].size(); idxOfNeuron++) {
			layers[idxOfLayer].at(idxOfNeuron)->WeightsUpdate();
		}
	}
}

void NeuralNetwork::ShowNetworkState()
{
	std::cout << std::endl;
	for (int indOfLayer = 0; indOfLayer < layers.size(); indOfLayer++) {
		std::cout << "Layer index: " << indOfLayer << std::endl;
		for (int indOfNeuron = 0; indOfNeuron < layers[indOfLayer].size(); indOfNeuron++) {
			std::cout << "  Neuron index: " << indOfNeuron << std::endl;
			layers[indOfLayer].at(indOfNeuron)->GetStatus();
		}
		if (indOfLayer < biasLayer.size()) {
			std::cout << "  Bias: " << std::endl;
			biasLayer[indOfLayer]->GetStatus();
		}
	}
}