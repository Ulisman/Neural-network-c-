#include "model.hpp"
#include "dense.hpp"
#include "activations.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <memory>

using namespace std;


Model::Model(){
    this->layers = vector<std::unique_ptr<Layer>>();
}

/**
 * @brief Adds a layer to the model
 * 
 * @param layer_ptr A pointer to a Layer object.
*/
void Model::addLayer(std::unique_ptr<Layer> layer_ptr){
    this->layers.push_back(std::move(layer_ptr));
}

/**
 * @brief Trains the model using the provided dataset over a specified number of epochs.
 *
 * @param input The input data as an Eigen::MatrixXf, where each row is a sample.
 * @param labels The one-hot encoded labels as an Eigen::MatrixXf, where each row corresponds to a sample's label.
 * @param epochs The number of training epochs.
 * @param lr The learning rate used for updating model parameters.
 */
void Model::fit(Eigen::MatrixXf& input, Eigen::MatrixXf& labels, int epochs, float lr){
    int batchSize = 1000; //hardcoded for now...
    int numSamples = input.rows();
    int numBatches = (numSamples + batchSize - 1) / batchSize;

    Eigen::MatrixXf modelOutput;
    Eigen::MatrixXf batchInputs;
    Eigen::MatrixXf batchLabels;
    for (int epoch = 0; epoch < epochs; epoch++){
        float loss;
        for (int batch = 0; batch < numBatches; batch++) {
            //calculate the batch
            int startIdx = batch * batchSize;
            int endIdx = std::min(startIdx + batchSize, numSamples);

            batchInputs = input.block(startIdx, 0, endIdx - startIdx, input.cols());
            batchLabels = labels.block(startIdx, 0, endIdx - startIdx, labels.cols());

            //forward pass
            modelOutput = this->forwardPass(batchInputs); 

            //calculate loss
            Eigen::MatrixXf predictionError = modelOutput.binaryExpr(batchLabels, std::function<float(float, float)>(activations::categoricalCrossentropy));
            loss = predictionError.mean();

            //backpropagation
            this->backprop(modelOutput, batchLabels, lr);
        }

        cout << "Loss for epoch " << epoch + 1 << ": " << loss << endl;
        this->evaluate(modelOutput, batchLabels, false);
    }
}

/**
 * @brief Preforms a forward pass on the input data.
 * 
 * @param inputs The input data as an Eigen::MatrixXf. Each row represents a sample.
 * @return Eigen::MatrixXf The output of the model as an Eigen::MatrixXf. Each row represents a sample.
*/
Eigen::MatrixXf Model::forwardPass(Eigen::MatrixXf& inputs){
    Eigen::MatrixXf outputs = inputs;
    for (auto& layer : this->layers){ //auto& — automatically detect the type (Layer) as a reference
        outputs = layer->forward(outputs);
    }
    return outputs;
}

/**
 * @brief Preforms a backpropagation pass on the training data and updates the model parameters.
 * 
 * @param input The input data as an Eigen::MatrixXf. Each row represents a sample.
 * @param labels The labels for the input data as an Eigen::MatrixXf. Each row is the one-hot encoded label for a sample.
 * @param epochs The number of epochs to run for training.
 * @param lr The learning rate for weight updates during backpropagation.
 */
void Model::backprop(Eigen::MatrixXf& modelOutput, Eigen::MatrixXf& labels, float& lr){
    //calculate gradients and deltas for output layer
    Eigen::MatrixXf outputDeltas = modelOutput - labels;
    this->layers.end()[-1]->setDeltas(outputDeltas);
    Eigen::MatrixXf outputGradients = this->layers.end()[-2]->getOutputActivations().transpose() * outputDeltas;
    this->layers.end()[-1]->setGradients(outputGradients);

    //calculate gradients for remaining layers (except Input layer)
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
    }

    //update weights for each layer in the model
    for (auto& layer: this->layers){
        layer->applyGradients(lr);
    }
}

/**
 * @brief Makes predictions on the input data.
 * 
 * @param batch The input data as an Eigen::MatrixXf. Each row represents a sample.
*/
Eigen::MatrixXf Model::predict(Eigen::MatrixXf& batch){
    return this->forwardPass(batch);
}

void Model::evaluate(Eigen::MatrixXf predictions, Eigen::MatrixXf& labels, bool isRealEvaluation){
    if (isRealEvaluation){
        predictions = this->predict(predictions); //overwrites the input
    }

    //argmax predictions
    Eigen::VectorXf predictionArgmax(predictions.rows());
    int idx;
    for (int i = 0; i < predictions.rows(); i++){
        predictions.row(i).maxCoeff(&idx);
        predictionArgmax(i) = idx;
    }

    //argmax labels
    int idxLabel;
    Eigen::VectorXf labelsArgmax(labels.rows());
    for (int i = 0; i < labels.rows(); i++){
        labels.row(i).maxCoeff(&idxLabel);
        labelsArgmax(i) = idxLabel;
    }
    
    //compute accuracy
    Eigen::ArrayXXf elementwiseCompare = (predictionArgmax.array() == labelsArgmax.array()).cast<float>();
    float accuracy = (elementwiseCompare.sum()/predictions.rows()) * 100;
    cout << "Accuracy: " << accuracy << "%" << endl;
}



