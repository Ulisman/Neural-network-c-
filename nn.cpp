#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace Eigen;
void loadData();

int main(){ 
    //
    Matrix <float, 3, 3> matrixA;
    matrixA.setZero();
    cout << matrixA <<endl;

    loadData();
}


void loadData(){
    string pathToData = "/Users/ulrikisdahl/Desktop/machine learning/mnist/trainingSet/trainingSet/0";
    int BATCH_SIZE = 4;  

    vector<cv::String> fn;
    cv::glob("/Users/ulrikisdahl/Desktop/machine learning/mnist/trainingSet/trainingSet/0/*jpg", fn, false);
    cout << fn.size() << endl;
    cout << fn[1] << endl;

    //cv::Mat img = cv::imread(fn[1], cv::IMREAD_GRAYSCALE);

    for (int i = 0; i < BATCH_SIZE; i++){
        cv::Mat img = cv::imread(fn[i], cv::IMREAD_GRAYSCALE);
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(8,8));

        VectorXf eigenImage(8*8);
        for (int i = 0; i < resized.rows; i++){
            for (size_t j = 0; j < resized.cols; j++){
                eigenImage(i*8 + j) = resized.at<uchar>(i,j);
            }
        }
    }

};  




