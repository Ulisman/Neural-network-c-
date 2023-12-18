#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace Eigen;
MatrixXf loadData();
float sigmoid(float);
float sigmoidDerivative(float);
float binaryCrossentropy(float, float);
float binaryError(float, float);


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
        NeuralNetwork(int batchSize, int inputHeight, int inputWidth, int numHiddenLayer1, int numHiddenLayer2, float lr){
            this->lr = lr;
            this->batchSize = batchSize;
            this->inputHeight = inputHeight;
            this->inputWidth = inputWidth;
            this->numHiddenLayer1 = numHiddenLayer1;
            this->numHiddenLayer2 = numHiddenLayer2;

            this->weights1 = MatrixXf::Random(this->inputHeight*this->inputWidth, numHiddenLayer1);
            this->weights2 = MatrixXf::Random(this->numHiddenLayer1, this->numHiddenLayer2);
            this->weights3 = MatrixXf::Random(this->numHiddenLayer2, 1); //hardcoded to 1 because of binary classification
            this->labels = VectorXf::Random(inputHeight);
        }

        void forwardAndBackward(MatrixXf& input){ //Input: matrise med shape (num_images, image_vector_length)
            Matrix<float, Dynamic, Dynamic> output1 = input * this->weights1;
            Matrix<float , Dynamic, Dynamic> activation1 = output1.unaryExpr(std::function<float(float)>(sigmoid)); //apply sigmoid activation function to the outputs of first hidden layer

            Matrix<float, Dynamic, Dynamic> output2 = activation1 * this->weights2;
            Matrix<float, Dynamic, Dynamic> activation2 = output2.unaryExpr(std::function<float(float)>(sigmoid));

            Matrix<float, Dynamic, Dynamic> output3 = activation2 * this->weights3; //raw logits
            Matrix<float, Dynamic, Dynamic> activation3 = output3.unaryExpr(std::function<float(float)>(sigmoid)); 

            //calculate prediction error
            this->error = activation3.binaryExpr(this->labels, std::function<float(float, float)>(binaryCrossentropy)); //MatrixXf is compatible with return type of 'binaryxpr'

            //calculate the deltas and gradients for output layer
            MatrixXf deltaOut = this->error.cwiseProduct(output3.unaryExpr(std::function<float(float)>(sigmoidDerivative)));
            MatrixXf gradientsW3 = output2.transpose() * deltaOut;

            //gradients for weights2
            MatrixXf errorHidden2 = deltaOut * this->weights3.transpose(); //(4, 1) * (4, 1).T
            MatrixXf deltaHidden2 = errorHidden2.cwiseProduct(output2.unaryExpr(std::function<float(float)>(sigmoidDerivative))); // (4,4)
            MatrixXf gradientsW2 = output1.transpose() * deltaHidden2;

            //gradients for weights1 
            MatrixXf errorHidden1 = deltaHidden2 * this->weights2.transpose(); // (4,4) * (4,4).T
            MatrixXf deltaHidden1 = errorHidden1.cwiseProduct(output1.unaryExpr(std::function<float(float)>(sigmoidDerivative))); //(4,4)
            MatrixXf gradientsW1 = input.transpose() * deltaHidden1; //(8, 64).T * (8, 4)

            //update weights
            this->weights3 -= this->lr*gradientsW3;
            this->weights2 -= this->lr*gradientsW2;
            this->weights1 -= this->lr*gradientsW1;
        };

    private:
        float lr;
        int batchSize;
        int inputHeight;
        int inputWidth;
        int numHiddenLayer1;
        int numHiddenLayer2;
        MatrixXf error;
        MatrixXf weights1;
        MatrixXf weights2;
        MatrixXf weights3;
        VectorXf labels;
};

float sigmoid(float elem){
    return 1 / (1 + std::exp(-elem));
}

float sigmoidDerivative(float elem){
    return sigmoid(elem) * (1 - sigmoid(elem));
};

float binaryCrossentropy(float elem, float target){       
    float loss =  -(target * std::log(elem) + (1 - target) * std::log(1 - elem));
    return loss;
};

float binaryError(float pred, float target){
    return pred - target;
}


int main(){ 
    //
    Matrix <float, 3, 3> matrixA;
    matrixA.setZero();
    cout << matrixA <<endl;

    MatrixXf train = loadData();
    
    NeuralNetwork nn(8, 8, 8, 4, 4, 0.001f);
    nn.forwardAndBackward(train);

}





