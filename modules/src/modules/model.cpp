#include "model.hpp"
#include "dense.hpp"
#include "activations.hpp"
#include <iostream>
#include <Eigen/Dense>

using namespace std;


Model::Model(string lossFunction, int batchSize){
    this->layers = vector<Dense>();
    this->lossFunction = lossFunction;
    this->batchSize = batchSize;
}

void Model::addLayer(Dense layer){
    this->layers.push_back(layer);
}

void Model::fit(Eigen::MatrixXf& input, Eigen::VectorXf& labels){
    //forward pass
    Eigen::MatrixXf modelOutput = this->forwardPass(input);

    //calculate error
    Eigen::MatrixXf error = modelOutput.binaryExpr(labels, std::function<float(float, float)>(activations::binaryCrossentropy));

    //calculate loss
    float loss = error.mean();
    cout << "Loss for current epoch: " << loss << endl;

    //backpropagation
    Eigen::MatrixXf outputDeltas = error.cwiseProduct(modelOutput.unaryExpr(std::function<float(float)>(activations::sigmoidDerivative)));
    Eigen::MatrixXf outputGradients = this->layers.end()[-2].getRawOutput().transpose() * outputDeltas; //dot product of sigmoid derivative of the last hidden layer with the deltas of the output
    //this->layers.end()
}

Eigen::MatrixXf Model::forwardPass(Eigen::MatrixXf& inputs){
    Eigen::MatrixXf outputs = inputs;
    for (auto& layer : this->layers){
        outputs = layer.forward(outputs);
    }
    return outputs;
}

void Model::backprop(){

}



