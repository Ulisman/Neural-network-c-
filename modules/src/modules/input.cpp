#include "layer.hpp"
#include "input.hpp"
#include <iostream>
#include <Eigen/Dense>

using namespace std;


Input::Input(Eigen::MatrixXf input, string name){
    this->name = name;
}

Eigen::MatrixXf Input::getOutputActivations() const{ 
    return this->inputs;
}

Eigen::MatrixXf Input::forward(const Eigen::MatrixXf& input) {
    this->inputs = input;
    return inputs;
}


string Input::getLayerName() const{
    return this->name;
}

Eigen::MatrixXf Input::getDeltas() const{
    throw std::logic_error("getDeltas not applicable for Input layer");
}

Eigen::MatrixXf Input::getWeights() const{
    throw std::logic_error("getWeights not applicable for Input layer");
}


Eigen::MatrixXf Input::getSigmoidDerivative() const{
    throw std::logic_error("getSigmoidDerivative not applicable for Input layer");
}


void Input::setDeltas(Eigen::MatrixXf& deltas){
    
}

void Input::setGradients(Eigen::MatrixXf& gradients){

}

void Input::applyGradients(float){

}


