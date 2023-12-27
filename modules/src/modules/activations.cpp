#include "activations.hpp"
#include <iostream>
using namespace std;

float activations::sigmoid(float x){
    return 1/(1+std::exp(-x));
}

float activations::sigmoidDerivative(float x){
    return activations::sigmoid(x) * (1-activations::sigmoid(x));
}

Eigen::MatrixXf activations::softmax(Eigen::MatrixXf logits){
    Eigen::MatrixXf logits_exp = logits.unaryExpr([](float x){return std::exp(x);});
    Eigen::VectorXf sum_exp_rowwise = logits_exp.rowwise().sum();
    return logits_exp.array().colwise() / sum_exp_rowwise.array();
}

float activations::binaryCrossentropy(float pred, float target){
    const float epsilon = 1e-7f; //epsilon for clipping
    pred = (pred < epsilon) ? epsilon : pred; // Clamp lower bound
    pred = (pred > 1.0f - epsilon) ? 1.0f - epsilon : pred; // Clamp upper bound
    //pred = std::clamp(pred, epsilon, 1.0f - epsilon); //returns epsilon if pred is lower, or 1.0f-epsilon if pred is higher than 1.0f-epsilon
    return -target * std::log(pred) - (1-target) * std::log(1-pred);
}

float activations::categoricalCrossentropy(float pred, float target){
    const float epsilon = 1e-7f; 
    pred = std::max(pred, epsilon);
    return -target * std::log(pred);
}

Eigen::MatrixXf activations::oneHotEncode(Eigen::VectorXf& labels, int numClasses){
    Eigen::MatrixXf oneHotEncodedMatrix(labels.size(), numClasses);
    for (int i = 0; i < labels.size(); i++){
        oneHotEncodedMatrix.row(i) = Eigen::VectorXf::Zero(numClasses);
        oneHotEncodedMatrix(i, static_cast<int>(labels[i])) = 1.0f;
    }
    return oneHotEncodedMatrix;
}



// Eigen::MatrixXf activations::sparseCategoricalCrossentropy(Eigen::MatrixXf logits){
//     Eigen::VectorXf maxCoeffs = logits.rowwise().maxCoeff();
//     Eigen::MatrixXf sparseCategoricalCrossentropy = logits.unaryExpr([&](float x){return activations::categoricalCrossentropy(x, maxCoeffs[0]);});
//     return sparseCategoricalCrossentropy;
// }



