#include "dense.hpp"
#include "activations.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <random>
using namespace std;


Dense::Dense(int neurons, string activationFunction, string layerName){ //Dense::Dense -> Dense constructor belongs to Dense class
    this->neurons = neurons;
    this->isInitialized = false;
    this->activationFunction = activationFunction;
    this->layerName = layerName;
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
    //He weight initialization
    static std::default_random_engine generator(static_cast<unsigned int>(time(0)));
    float k = std::sqrt(1/(float)cols);
    std::uniform_real_distribution<float> distribution(-k, k);
    this->weights = Eigen::MatrixXf::Zero(rows, cols).unaryExpr([&](float x){return distribution(generator);});
}

Eigen::MatrixXf Dense::forward(const Eigen::MatrixXf& input){
    if (!this->isInitialized){
        this->isInitialized = true;
        this->initializeWeights(input.cols(), this->neurons);
    }

    this->rawOutputs = input * this->weights; //logits
    
    if(this->activationFunction == "sigmoid"){
        this->outputActivations = this->rawOutputs.unaryExpr(std::function<float(float)>(activations::sigmoid));
    } 
    else if (this->activationFunction == "softmax"){
        this->outputActivations = activations::softmax(this->rawOutputs);
    }
    else {
        return this->rawOutputs;
    }

    return this->outputActivations;
}

Eigen::MatrixXf Dense::getWeights() const{
    return this->weights;
}

Eigen::MatrixXf Dense::getSigmoidDerivative() const{
    return this->rawOutputs.unaryExpr(std::function<float(float)>(activations::sigmoidDerivative));
}

void Dense::setGradients(Eigen::MatrixXf& gradients) {
    this->gradients = gradients;
}

void Dense::setDeltas(Eigen::MatrixXf& deltas) {
    this->outputDeltas = deltas;
}

Eigen::MatrixXf Dense::getDeltas() const{
    return this->outputDeltas;
}

void Dense::applyGradients(float lr){
    this->weights = this->weights - lr * this->gradients;
}

Eigen::MatrixXf Dense::getOutputActivations() const{
    return this->outputActivations;
}

string Dense::getLayerName() const{
    return this->layerName;
}

void Dense::backprop(){

}




