#include "model.hpp"
#include "dense.hpp"
#include "activations.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <memory>

using namespace std;


Model::Model(string lossFunction, int batchSize){
    this->layers = vector<std::unique_ptr<Layer>>();
    this->lossFunction = lossFunction;
    this->batchSize = batchSize;
}

void Model::addLayer(std::unique_ptr<Layer> layer_ptr){
    this->layers.push_back(std::move(layer_ptr));
}

void Model::fit(Eigen::MatrixXf& input, Eigen::VectorXf& labels, float lr){
    //forward pass
    Eigen::MatrixXf modelOutput = this->forwardPass(input);

    //calculate loss
    Eigen::MatrixXf error = modelOutput.binaryExpr(labels, std::function<float(float, float)>(activations::binaryCrossentropy));
    float loss = error.mean();
    cout << "Loss for current epoch: " << loss << endl;

    //backpropagation
    //start with deltas and gradients for output layer
    Eigen::MatrixXf outputDeltas = error.cwiseProduct(modelOutput.unaryExpr(std::function<float(float)>(activations::sigmoidDerivative))); //Hadamard product
    this->layers.end()[-1]->setDeltas(outputDeltas); //this->layers.end()[-1] is a unique_ptr to the last layer in the model
    Eigen::MatrixXf outputGradients = this->layers.end()[-2]->getOutputActivations().transpose() * outputDeltas; //dot product of activations of the last hidden layer with the deltas of the output
    this->layers.end()[-1]->setGradients(outputGradients);

    //calculate gradients for remaining layers
    Eigen::MatrixXf currentLayerError;
    Eigen::MatrixXf currentLayerDeltas;
    Eigen::MatrixXf currentLayerGradients;
    for(int i = 0; i < this->layers.size() - 2; i++){ 
        auto& currentLayer = this->layers.end()[-i - 2];
        auto& nextLayer = this->layers.end()[-i - 1]; // the layer that was added AFTER current layer to the model
        auto& previousLayer = this->layers.end()[-i - 3]; // the layer that was added BEFORE current layer to the model

        currentLayerError = nextLayer->getDeltas() * nextLayer->getWeights().transpose();

        currentLayerDeltas = currentLayerError.cwiseProduct(currentLayer->getSigmoidDerivative());
        currentLayer->setDeltas(currentLayerDeltas);

        currentLayerGradients = previousLayer->getOutputActivations().transpose() * currentLayerDeltas;
        currentLayer->setGradients(currentLayerGradients);
        cout << "prev for this iteration: " << i << previousLayer->getLayerName() << endl;
    }

    //update weights for each layer in the model
    for (auto& layer: this->layers){
        layer->applyGradients(lr);
    }

}

Eigen::MatrixXf Model::forwardPass(Eigen::MatrixXf& inputs){

    Eigen::MatrixXf outputs = inputs;
    for (auto& layer : this->layers){ //auto& — automatically detect the type (Layer) as a reference
        outputs = layer->forward(outputs);
    }
    return outputs;
}

void Model::backprop(){

}



