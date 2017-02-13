#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "NeuralNetwork.h"
#include "TrainAlgorithm.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
using namespace std;

//vector<double> downsample();
//void getFiles(map<vector<double>, int> &train);
void paint(vector<double> data);

int main(int argc, char **argv) {
	if (argc == 1) {
		cout << "--Create Neural Network" << endl;
		NeuralNetwork *net = new NeuralNetwork(784, 10, 2, 256);
		cout << "--Neural Network succsessfully created" << endl;

		ifstream trainStream("../../../../data/train.csv");
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
		while (i < 20000) {
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
		while (i < 5000) {
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
		cout << "--End of testing Neural Network" << endl;
		cout << "--Start saving network data" << endl;
		net->SaveParameters("output.txt");
		cout << "--End saving network data" << endl;
		trainStream.close();
		return 0;
	}
	else {
		cout << "--Create Neural Network" << endl;
		NeuralNetwork *net = new NeuralNetwork(argv[1]);
		cout << "--Neural Network succsessfully created" << endl;
		ifstream trainStream("../../../../data/train.csv");
		if (!trainStream) {
			cout << "error with opening file!";
			return -1;
		}

		cout << "--Start testing" << endl;
		int i = 0;
		int rightAnswers = 0;
		char buf[5000];
		while (i < 20000) {
			trainStream.getline(buf, 5000);
			i++;
		}
		i = 0;
		while (i < 5000) {
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

		char ready = 'Y';
		while (1) {
			cout << "Ready to play? [Y/N]: ";
			cin >> ready;
			if (ready != 'Y')
				break;

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

			paint(test);

			uchar playerAnswer;
			cout << "Your answer:\t";
			cin >> playerAnswer;

			uchar netResponse = net->GetNetResponse(test);
			cout << "Net response:\t" << (int)netResponse << endl;
			cout << "Right answer:\t" << (int)answer << endl;
			//if ((int)netResponse == (int)answer)
				//if((int)playerAnswer != (int)answer)
					//if((int)playerAnswer != (int)netResponse)
						//cout << "Ha-ha! The net smarter than you!" << endl;
			
		}
			return 0;
	}
}


void paint(vector<double> data) {
	using namespace cv;
	Mat img(28, 28, CV_8UC1);
	for (int i = 0; i < 784; i++)
		img.data[i] = data.at(i);
	Mat dst(70, 70, CV_8UC1);
	resize(img, dst, dst.size(), 0, 0, 3);
	
	const string kDstWindowName = "numeral";
	namedWindow(kDstWindowName, WINDOW_AUTOSIZE);
	imshow(kDstWindowName, dst);
	if (waitKey(30) >= 0) return;
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