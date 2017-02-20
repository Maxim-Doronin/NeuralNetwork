#include "NeuralNetwork.h"
#include "TrainAlgorithm.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>

using namespace std;

int main(int argc, char **argv) {
	cout << "--Create Neural Network" << endl;
	NeuralNetwork *net = new NeuralNetwork(argv[2]);
	cout << "--Neural Network succsessfully created" << endl;
	
	ifstream testStream(argv[1]);
	if (!testStream) {
		cout << "error with opening file!";
		return -1;
	}

	int testSize = atoi(argv[3]);
	
	cout << "--Start testing" << endl;
	int i = 0;
	char buf[5000];
	while (i < testSize) {
		testStream.getline(buf, 5000);
		i++;
	}
		
	i = 0;
	int rightAnswers = 0;
	while (i < testSize) {
		testStream.getline(buf, 5000);
		char *number = strtok(buf, ",");
		uchar answer = atoi(number);

		vector<double> test;
		uchar tmp;
		for (int j = 0; j < 784; j++) {
			number = strtok(NULL, ",");
			tmp = atoi(number);
			test.push_back(tmp);
		}
		if (net->GetNetResponse(test) == answer)
			rightAnswers++;
		int temp = answer;
		i++;
	}
		
	cout << "Rate " << rightAnswers / (float)i << endl;
	cout << "--End of testing Neural Network" << endl;

	return 0;
}