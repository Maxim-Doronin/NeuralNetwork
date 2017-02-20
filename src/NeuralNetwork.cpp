#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const int& _inputs, const int& _outputs,
	const int& _numOfHiddenLayers, const int& _hidden, const char* type)
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
			Neuron* outN = neuralFactory->CreateOutputNeuron(OutputNeuronsFunc);
			outputLayer.push_back(outN);
		}
		layers.push_back(outputLayer);

		//create hidden neurons
		for (int i = 0; i < _numOfHiddenLayers; i++) {
			std::vector<Neuron*> HiddenLayer;
			for (int j = 0; j < hidden; j++) {
				Neuron* hidden = neuralFactory->CreateHiddenNeuron(layers[0], OutputNeuronsFunc);
				HiddenLayer.push_back(hidden);
			}
			Neuron* hidN = neuralFactory->CreateInputNeuron(layers[0], InputNeuronsFunc);
			biasLayer.insert(biasLayer.begin(), hidN);
			layers.insert(layers.begin(), HiddenLayer);
		}

		//create input neurons
		for (int i = 0; i < inputs; i++) {
			Neuron* inpN = neuralFactory->CreateInputNeuron(layers[0], InputNeuronsFunc);
			inputLayer.push_back(inpN);
		}
		
		//create bias
		Neuron* biasN = neuralFactory->CreateInputNeuron(layers[0], InputNeuronsFunc);
		biasLayer.insert(biasLayer.begin(), biasN);
		layers.insert(layers.begin(), inputLayer);

		trainingAlgoritm->WeightsInitialization();
	}
}

NeuralNetwork::NeuralNetwork(const char* filename)
{
	std::ifstream input(filename);
	if (!input) {
		std::cout << "--Cannot open file" << std::endl;
		exit(1);
	}

	Function* OutputNeuronsFunc;
	Function* InputNeuronsFunc;
	std::vector<Neuron*> outputLayer;
	std::vector<Neuron*> inputLayer;
	neuralFactory = new PerceptronNeuronFactory;
	trainingAlgoritm = new Backpropagation(this);
	OutputNeuronsFunc = new Sigmoid;
	InputNeuronsFunc = new Linear;

	int layerNumber;
	input >> layerNumber;
	input >> inputs;
	input >> outputs;
	input >> hidden;

	//create output neurons
	for (int i = 0; i < outputs; i++) {
		Neuron* outN = neuralFactory->CreateOutputNeuron(OutputNeuronsFunc);
		outputLayer.push_back(outN);
	}
	layers.push_back(outputLayer);

	//create hidden neurons
	for (int i = layerNumber - 2; i > 0; i--) {
		std::vector<Neuron*> HiddenLayer;
		for (int j = 0; j < hidden; j++) {
			Neuron* hidN = neuralFactory->CreateHiddenNeuron(layers[0], OutputNeuronsFunc);
			double weight;
			for (int k = 0; k < layers[0].size(); k++) {
				input >> weight;
				hidN->GetOutputLinks().at(k)->SetWeigth(weight);
			}
			HiddenLayer.push_back(hidN);
		}
		layers.insert(layers.begin(), HiddenLayer);
	}

	//create input neurons
	for (int i = 0; i < inputs; i++) {
		Neuron *inpN = neuralFactory->CreateInputNeuron(layers[0], InputNeuronsFunc);
		double weight;
		for (int k = 0; k < layers[0].size(); k++) {
			input >> weight;
			inpN->GetOutputLinks().at(k)->SetWeigth(weight);
		}
		inputLayer.push_back(inpN);
	}
	layers.insert(layers.begin(), inputLayer);

	//create bias
	int biasNumber;
	input >> biasNumber;
	for (int i = 1; i < layerNumber; i++) {
		Neuron *biasN = neuralFactory->CreateInputNeuron(layers[i], InputNeuronsFunc);
		double weight;
		for (int k = 0; k < layers[i].size(); k++) {
			input >> weight;
			biasN->GetOutputLinks().at(k)->SetWeigth(weight);
		}
		biasLayer.push_back(biasN);
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

bool NeuralNetwork::Train(const std::vector<std::vector<double> >& data, 
	const std::vector<std::vector<double> >& target)
{
	bool flag = true;
	int iterations = 0;
	double oldMSE = 11;
	double newMSE;
	int repeats = 0;;
	while (flag && iterations < 15) {
		iterations++;
		std::cout << "--Start algorithm" << std::endl;
		for (int i = 0; i < data.size(); i++) {
			if (i % 100 == 0) {
				newMSE = this->GetMSE();
				std::cout << newMSE << std::endl;
				oldMSE = newMSE;
				this->ResetMSE();
			}
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

		for (int layerIdx = 0; layerIdx < layers.size() - 1; layerIdx++) {
			biasLayer[layerIdx]->Input(1);
			for (int neurIdx = 0; neurIdx < layers.at(layerIdx).size(); neurIdx++) {
				layers.at(layerIdx).at(neurIdx)->Activation();
			}
			biasLayer[layerIdx]->Activation();
		}

		double maxRes = -1;
		int answer;
		for (int ioutput = 0; ioutput < outputs; ioutput++) {
			double res = this->GetOutputLayer().at(ioutput)->Activation();
			if (res > maxRes) {
				maxRes = res;
				answer = ioutput;
			}
		}
		this->ResetSums();
		return answer;
	}
}

void NeuralNetwork::SetAlgorithm(TrainAlgorithm* _trainingAlgorithm)
{ 
	trainingAlgoritm = _trainingAlgorithm; 
}

void NeuralNetwork::SetNeuronFactory(NeuronFactory* neuronFactory)
{ 
	neuralFactory = neuronFactory; 
}

const double& NeuralNetwork::GetMinMSE()
{ 
	return minMSE; 
}

void NeuralNetwork::SetMinMSE(const double& _minMse)
{ 
	minMSE = _minMse; 
}

std::vector<Neuron *>&	NeuralNetwork::GetLayer(const int& idx)
{ 
	return layers[idx]; 
}

int NeuralNetwork::size()
{ 
	return (int)layers.size(); 
}

std::vector<Neuron*>&	NeuralNetwork::GetOutputLayer()
{ 
	return layers[layers.size() - 1]; 
}

std::vector<Neuron*>&	NeuralNetwork::GetInputLayer()
{ 
	return layers[0]; 
}

std::vector<Neuron*>& 	NeuralNetwork::GetBiasLayer()
{ 
	return biasLayer; 
}

void NeuralNetwork::UpdateWeights()
{
	for (int layerIdx = 0; layerIdx < layers.size(); layerIdx++) {
		for (int neurIdx = 0; neurIdx < layers[layerIdx].size(); neurIdx++) {
			layers[layerIdx].at(neurIdx)->WeightsUpdate();
		}
	}
}

void NeuralNetwork::ShakeWeights() {
	for (int layerIdx = 0; layerIdx < layers.size(); layerIdx++) {
		for (int neurIdx = 0; neurIdx < layers[layerIdx].size(); neurIdx++) {
			layers[layerIdx].at(neurIdx)->ShakeWeights();
		}
	}
}

void NeuralNetwork::ResetSums()
{
	for (int layerIdx = 0; layerIdx < layers.size(); layerIdx++)
		for (int neurIdx = 0; neurIdx < layers.at(layerIdx).size(); neurIdx++)
			layers.at(layerIdx).at(neurIdx)->ResetTotalSum();

	for (int layerIdx = 0; layerIdx < layers.size() - 1; layerIdx++)
		biasLayer[layerIdx]->ResetTotalSum();
}

void NeuralNetwork::AddMSE(double localMSE)
{ 
	meanSquaredError += localMSE; 
}

double NeuralNetwork::GetMSE()
{ 
	return meanSquaredError; 
}

void NeuralNetwork::ResetMSE()
{ 
	meanSquaredError = 0; 
}

void NeuralNetwork::ShowNetworkState()
{
	std::cout << std::endl;
	for (int layerIdx = 0; layerIdx < layers.size(); layerIdx++) {
		std::cout << "Layer index: " << layerIdx << std::endl;
		for (int neurIdx = 0; neurIdx < layers[layerIdx].size(); neurIdx++) {
			std::cout << "  Neuron index: " << neurIdx << std::endl;
			layers[layerIdx].at(neurIdx)->GetStatus();
		}
		if (layerIdx < biasLayer.size()) {
			std::cout << "  Bias: " << std::endl;
			biasLayer[layerIdx]->GetStatus();
		}
	}
}

void NeuralNetwork::SaveParameters(const char* filename)
{
	remove(filename);
	std::ofstream output(filename);

	output << this->size() << std::endl;
	output << this->inputs << std::endl;
	output << this->outputs << std::endl;
	output << this->hidden << std::endl;
	for (int i = this->size() - 2; i >= 0; i--) {
		int layerSize = (int)this->GetLayer(i).size();
		for (int j = 0; j < layerSize; j++) {
			Neuron* neuron = this->GetLayer(i).at(j);
			int linksNumber = (int)neuron->GetOutputLinks().size();
			for (int k = 0; k < linksNumber; k++) {
				NeuralLink* link = neuron->GetOutputLinks().at(k);
				double weight = link->GetWeigth();
				output << weight << ' ';
			}
			output << std::endl;
		}
	}

	int biasSize = (int)this->biasLayer.size();
	output << biasSize << std::endl;
	for (int i = 0; i < biasSize; i++) {
		Neuron* neuron = this->biasLayer.at(i);
		int linksNumber = (int)neuron->GetOutputLinks().size();
		for (int j = 0; j < linksNumber; j++) {
			NeuralLink* link = neuron->GetOutputLinks().at(j);
			double weight = link->GetWeigth();
			output << weight << ' ';
		}
		output << std::endl;
	}
}