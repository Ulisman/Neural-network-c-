#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace Eigen;
MatrixXf loadData();

float sigmoid(float);
float binaryCrossentropy(float, int);


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

        VectorXf eigenImage(8*8);
        for (int i = 0; i < resized.rows; i++){
            for (size_t j = 0; j < resized.cols; j++){
                eigenImage(i*8 + j) = resized.at<uchar>(i,j);
            }
        }
        batchMatrix.row(i) = eigenImage;
    }

    return batchMatrix;
};  


class NeuralNetwork{

    public:
        NeuralNetwork(int batchSize= 4, int inputHeight = 8, int inputWidth = 8, int numHiddenLayer1 = 4, int numHiddenLayer2 = 4){
            this->batchSize = batchSize;
            this->inputHeight = inputHeight;
            this->inputWidth = inputWidth;
            this->numHiddenLayer1 = numHiddenLayer1;
            this->numHiddenLayer2 = numHiddenLayer2;

            this->inputWeights = MatrixXf::Random(this->inputHeight*this->inputWidth, numHiddenLayer1);
            this->hidden1Weigths = MatrixXf::Random(this->numHiddenLayer1, this->numHiddenLayer2);
            this->hidden2Weights = MatrixXf::Random(this->numHiddenLayer2, 1); //hardcoded to 1 because of binary classification

            this->applySigmoid = sigmoid;
        }

        void forward(MatrixXf input){ //Input: matrise med shape (num_images, image_vector_length)
            Matrix<float, Dynamic, Dynamic> output1 = input * this->inputWeights;
            Matrix<float , Dynamic, Dynamic> activation1 = output1.unaryExpr(std::function<float(float)>(sigmoid)); //apply sigmoid activation function to the outputs of first hidden layer

            Matrix<float, Dynamic, Dynamic> output2 = activation1 * this->hidden1Weigths;
            Matrix<float, Dynamic, Dynamic> activation2 = output2.unaryExpr(std::function<float(float)>(sigmoid));

            Matrix<float, Dynamic, Dynamic> output3 = activation2 * this->hidden2Weights; //raw logits
            Matrix<float, Dynamic, Dynamic> activation3 = output3.unaryExpr(std::function<float(float)>(sigmoid)); 

            float sum = 0;
            for (int i = 0; i < activation3.rows(); i++){
                sum += binaryCrossentropy(activation3(i), 0);
            }
            float loss = sum / activation3.rows();
            
            cout << loss << endl;
        };

        void backward(){
            
        };

    private:
        int batchSize;
        int inputHeight;
        int inputWidth;
        int numHiddenLayer1;
        int numHiddenLayer2;
        MatrixXf inputWeights;
        MatrixXf hidden1Weigths;
        MatrixXf hidden2Weights;
        std::function<float(float)> applySigmoid;

};

float sigmoid(float elem){
    return 1 / (1 + std::exp(-elem));
}

float sigmoidDerivative(float elem){
    return 1.0;
};

float binaryCrossentropy(float elem, int target){       
    cout << "ELEM" << endl;
    cout << elem << endl;
    float loss =  -(target * std::log(elem) + (1 - target) * std::log(1 - elem));
    return loss;
};



int main(){ 
    //
    Matrix <float, 3, 3> matrixA;
    matrixA.setZero();
    cout << matrixA <<endl;

    MatrixXf train = loadData();
    
    NeuralNetwork nn(4, 8, 8, 4, 4);
    nn.forward(train);

}

