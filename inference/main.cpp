#include <iostream>
#include <modules/dense.hpp>
#include <modules/model.hpp>
#include <modules/input.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
using namespace std;

Eigen::MatrixXf loadData(){
    string pathToData = "/Users/ulrikisdahl/Desktop/machine learning/mnist/trainingSet/trainingSet/0";
    int BATCH_SIZE = 8;  

    vector<cv::String> fn;
    cv::glob("/Users/ulrikisdahl/Desktop/machine learning/mnist/trainingSet/trainingSet/0/*jpg", fn, false);
    cout << fn.size() << endl;
    cout << fn[1] << endl;

    Eigen::MatrixXf batchMatrix(BATCH_SIZE, 8*8);
    for (int i = 0; i < BATCH_SIZE; i++){
        cv::Mat img = cv::imread(fn[i], cv::IMREAD_GRAYSCALE);
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(8,8));

        Eigen::VectorXf eigenImage(8*8); //one dimensional vector with 64 elements
        for (int i = 0; i < resized.rows; i++){
            for (size_t j = 0; j < resized.cols; j++){
                eigenImage[i*8 + j] = resized.at<uchar>(i,j);
            }
        }
        batchMatrix.row(i) = eigenImage;
    }

    return batchMatrix;
};  



int main(){
    Eigen::MatrixXf data = loadData();
    cout << data.rows() << endl;
    cout << data.cols() << endl;

    Model model = Model("binary crossentropy", 8);

    //Input inp(data, "inputLayer"); 
    model.addLayer(std::make_unique<Input>(data, "inputLayer")); 

    //Dense dense(64, "sigmoid", "hidden1");
    //model.addLayer(dense);
    model.addLayer(std::make_unique<Dense>(64, "sigmoid", "hidden1"));

    // Dense dense2(64, "sigmoid", "hidden2");
    // model.addLayer(dense2);
    model.addLayer(std::make_unique<Dense>(64, "sigmoid", "hidden2"));

    // Dense dense3(1, "sigmoid", "output"); //biary cross entropy expects a probability as input
    // model.addLayer(dense3);
    model.addLayer(std::make_unique<Dense>(1, "sigmoid", "output"));

    Eigen::VectorXf test(8);
    test << 0, 0, 0, 0, 0, 0, 0, 0; 
    cout << test.rows() << endl;

    model.fit(data, test, 0.001f);

    //compute loss

    //use loss to move backwards and compute gradients (backprop)

    cout << "main working" << endl;
    return 0;
}