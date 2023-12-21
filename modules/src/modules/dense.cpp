//#include <modules/dense.hpp>
#include "dense.hpp"
#include "activations.hpp"
#include <iostream>
#include <Eigen/Dense>
using namespace std;


Dense::Dense(int neurons, string activationFunction){ //Dense::Dense -> Dense constructor belongs to Dense class
    this->neurons = neurons;
    this->isInitialized = false;
    this->activationFunction = activationFunction;
}

/**
 * @brief Initializes the weight matrix for the Dense layer.
 *
 * @param rows The number of rows in the weight matrix, corresponding to the 
 *             number of output neurons from previous layer or number of input features.
 * @param cols The number of columns in the weight matrix, corresponding to the
 *             number of neurons in the Dense layer.
 */
void Dense::initializeWeights(const int& rows, const int& cols){
    std::srand((unsigned int) time(0));
    this->weights = Eigen::MatrixXf::Random(rows, cols);
    cout << this->weights << endl;
}

Eigen::MatrixXf Dense::forward(const Eigen::MatrixXf& input){
    if (!this->isInitialized){
        this->isInitialized = true;
        this->initializeWeights(input.cols(), this->neurons);
    }

    this->rawOutputs = input * this->weights;
    Eigen::MatrixXf activations; //We don't need the activations for backprop (only raw outputs) so we can use local variable here
    if(this->activationFunction == "sigmoid"){
        activations = this->rawOutputs.unaryExpr(std::function<float(float)>(activations::sigmoid));
    } else {
        return this->rawOutputs;
    }

    return activations;
}

Eigen::MatrixXf Dense::getWeights(){
    return this->weights;
}

Eigen::MatrixXf Dense::getRawOutput(){
    return this->rawOutputs;
}

Eigen::MatrixXf Dense::getSigmoidDerivative(){
    return this->rawOutputs.unaryExpr(std::function<float(float)>(activations::sigmoidDerivative));
}

void Dense::backprop(){

}




