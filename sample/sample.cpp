#include "NeuralNetwork.h"
#include "TrainAlgorithm.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>

using namespace std;

//vector<double> downsample();
//void getFiles(map<vector<double>, int> &train);

int main(int argc, char **argv) {
	cout << "--Create Neural Network" << endl;
	NeuralNetwork *net = new NeuralNetwork(784, 10, 2, 800);
	cout << "--Neural Network succsessfully created" << endl;

	ifstream trainStream("../../../data/train.csv");
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
	while (i < 10000) {
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
	int rightAnswers;
	while (i < 2000) {
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
		//cout << temp << endl;
		i++;
	}
	cout << "Rate " << rightAnswers / (float)i << endl;
	trainStream.close();
	return 0;
}

/*
void getFiles(map<vector<double>, int> &train, string foldername) {
	WIN32_FIND_DATAW wfd;

	string fullpath = "C:\\Users\\asus\\BackProc\\data\\train\\" + foldername + "\\*";
	wstring stemp = std::wstring(fullpath.begin(), fullpath.end());
	LPCWSTR sw = stemp.c_str();
	HANDLE const hFind = FindFirstFileW(sw, &wfd);
	setlocale(LC_ALL, "");

	string filename;
	if (INVALID_HANDLE_VALUE != hFind)
	{
		do
		{
			char ch[260];
			char DefChar = ' ';
			WideCharToMultiByte(CP_ACP, 0, &wfd.cFileName[0], -1, ch, 260, &DefChar, NULL);
			filename.assign(ch);
			Mat src = imread("../../../data/" + foldername + "/" + filename);
			cvtColor(src, src, CV_BGR2GRAY);
			for (int i = 0; i < src.rows * src.cols; i++) {


			}


		} while (NULL != FindNextFileW(hFind, &wfd));

		FindClose(hFind);
	}
}
*/