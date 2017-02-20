#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const int& _inputs, const int& _outputs, const int& _numOfHiddenLayers,
			const int& _hidden, const char* type)
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
	for (int i = 0; i < outputs; i++) {
		outputLayer.push_back(neuralFactory->CreateOutputNeuron(OutputNeuronsFunc));
	}
	layers.push_back(outputLayer);

	for (int i = layerNumber - 2; i > 0; i--) {
		std::vector<Neuron*> HiddenLayer;
		for (int j = 0; j < hidden; j++) {
			Neuron* hiddenNeuron = neuralFactory->CreateHiddenNeuron(layers[0], OutputNeuronsFunc);
			double weight;
			for (int k = 0; k < layers[0].size(); k++) {
				input >> weight;
				hiddenNeuron->GetOutputLinks().at(k)->SetWeigth(weight);
			}
			HiddenLayer.push_back(hiddenNeuron);
		}
		layers.insert(layers.begin(), HiddenLayer);
	}

	for (int i = 0; i < inputs; i++) {
		Neuron *inputNeuron = neuralFactory->CreateInputNeuron(layers[0], InputNeuronsFunc);
		double weight;
		for (int k = 0; k < layers[0].size(); k++) {
			input >> weight;
			inputNeuron->GetOutputLinks().at(k)->SetWeigth(weight);
		}
		inputLayer.push_back(inputNeuron);
	}
	layers.insert(layers.begin(), inputLayer);

	int biasNumber;
	input >> biasNumber;

	for (int i = 1; i < layerNumber; i++) {
		Neuron *biasNeuron = neuralFactory->CreateInputNeuron(layers[i], InputNeuronsFunc);
		double weight;
		for (int k = 0; k < layers[i].size(); k++) {
			input >> weight;
			biasNeuron->GetOutputLinks().at(k)->SetWeigth(weight);
		}
		biasLayer.push_back(biasNeuron);
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
				/*
				if (fabs(oldMSE - newMSE) < 0.1) {
					repeats++;
					if (repeats >= 3) {
						this->ShakeWeights();
						std::cout << "shaked" << std::endl;
						repeats = 0;
					}
				}
				else
					repeats = 0;
					*/
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

		for (int numOfLayers = 0; numOfLayers < layers.size() - 1; numOfLayers++) {
			biasLayer[numOfLayers]->Input(1);

			for (int indexOfData = 0; indexOfData < layers.at(numOfLayers).size(); indexOfData++) {
				layers.at(numOfLayers).at(indexOfData)->Activation();
			}

			biasLayer[numOfLayers]->Activation();
		}
		double maxRes = -1;
		int answer;
		//std::cout << "Net response is: { ";
		for (int ioutput = 0; ioutput < outputs; ioutput++) {
			double res = this->GetOutputLayer().at(ioutput)->Activation();
			if (res > maxRes) {
				maxRes = res;
				answer = ioutput;
			}
			//std::cout.precision(4);
			//std::cout.setf(std::ios::fixed);
			//std::cout << res << "\t";
			//netResponse.push_back(res);
		}
		//std::cout << "|| " << answer << " } ";
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

unsigned int NeuralNetwork::size()
{ 
	return layers.size(); 
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
	for (int idxOfLayer = 0; idxOfLayer < layers.size(); idxOfLayer++) {
		for (int idxOfNeuron = 0; idxOfNeuron < layers[idxOfLayer].size(); idxOfNeuron++) {
			layers[idxOfLayer].at(idxOfNeuron)->WeightsUpdate();
		}
	}
}

void NeuralNetwork::ShakeWeights() {
	for (int idxOfLayer = 0; idxOfLayer < layers.size(); idxOfLayer++) {
		for (int idxOfNeuron = 0; idxOfNeuron < layers[idxOfLayer].size(); idxOfNeuron++) {
			layers[idxOfLayer].at(idxOfNeuron)->ShakeWeights();
		}
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


void NeuralNetwork::SaveParameters(const char* filename)
{
	remove(filename);
	std::ofstream output(filename);

	output << this->size() << std::endl;
	output << this->inputs << std::endl;
	output << this->outputs << std::endl;
	output << this->hidden << std::endl;
	for (int i = this->size() - 2; i >= 0; i--) {
		int layerSize = this->GetLayer(i).size();
		for (int j = 0; j < layerSize; j++) {
			Neuron* neuron = this->GetLayer(i).at(j);
			int linksNumber = neuron->GetOutputLinks().size();
			for (int k = 0; k < linksNumber; k++) {
				NeuralLink* link = neuron->GetOutputLinks().at(k);
				double weight = link->GetWeigth();
				output << weight << ' ';
			}
			output << std::endl;
		}
	}

	int biasSize = this->biasLayer.size();
	output << biasSize << std::endl;
	for (int i = 0; i < biasSize; i++) {
		Neuron* neuron = this->biasLayer.at(i);
		int linksNumber = neuron->GetOutputLinks().size();
		for (int j = 0; j < linksNumber; j++) {
			NeuralLink* link = neuron->GetOutputLinks().at(j);
			double weight = link->GetWeigth();
			output << weight << ' ';
		}
		output << std::endl;
	}
}