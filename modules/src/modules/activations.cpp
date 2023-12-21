#include "activations.hpp"
#include <iostream>
using namespace std;

float activations::sigmoid(float x){
    return 1/(1+std::exp(-x));
}

float activations::sigmoidDerivative(float x){
    return activations::sigmoid(x) * (1-activations::sigmoid(x));
}

float activations::binaryCrossentropy(float pred, float target){
    return -target * std::log(pred) - (1-target) * std::log(1-pred);
}