#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "NeuralNetwork.h"
#include "TrainAlgorithm.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
using namespace std;


int main(int argc, char **argv) {
	cout << "--Create Neural Network" << endl;
	NeuralNetwork *net = new NeuralNetwork(784, 10, 2, 256);
	cout << "--Neural Network succsessfully created" << endl;

	ifstream trainStream(argv[1]);
	if (!trainStream) {
		cout << "error with opening file!";
		return -1;
	}

	cout << "--Start reading train data" << endl;
	int i = 0;
	vector<vector<double> > traindata;
	vector<vector<double> > targetdata;
	uchar tmp;
	char buf[5000];
	int trainSize = atoi(argv[2]);
	while (i < trainSize) {
		trainStream.getline(buf, 5000);
		char *number = strtok(buf, ",");

		vector<double> target;
		uchar tmp = atoi(number);
		for (int k = 0; k < 10; k++) {
			if (k == tmp)
				target.push_back(1.0);
			else
				target.push_back(0.0);
		}
		targetdata.push_back(target);

		vector<double> train;
		for (int j = 0; j < 784; j++) {
			number = strtok(NULL, ",");
			tmp = atoi(number);
			train.push_back(tmp);
		}
		traindata.push_back(train);
		i++;
	}
	cout << "--Reading train data ended" << endl;
	cout << "--Start training Neural Network" << endl;
	net->Train(traindata, targetdata);
	cout << "--Training Neural Network ended" << endl;
	cout << "--Succsessfuly trained" << endl;
	cout << "--Start testing" << endl;


	i = 0;
	int rightAnswers = 0;
	while (i < 1000) {
		trainStream.getline(buf, 5000);
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
	cout << "--Start saving network data" << endl;
	net->SaveParameters("output.txt");
	cout << "--End saving network data" << endl;
	trainStream.close();
	return 0;
}