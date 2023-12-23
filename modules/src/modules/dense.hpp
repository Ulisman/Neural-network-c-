#ifndef DENSE_HPP
#define DENSE_HPP
#include <Eigen/Dense>
#include "layer.hpp"
using namespace std;


class Dense : public Layer {
public:
    Dense(int, string, string);

    string getLayerName() const override;
    Eigen::MatrixXf forward(const Eigen::MatrixXf&) override; //move to private later
    Eigen::MatrixXf getWeights() const override;
    Eigen::MatrixXf getDeltas() const override;
    //Eigen::MatrixXf getRawOutput() const override;
    Eigen::MatrixXf getSigmoidDerivative() const override;
    Eigen::MatrixXf getOutputActivations() const override;
    void setGradients(Eigen::MatrixXf&) override;
    void setDeltas(Eigen::MatrixXf&) override;
    void applyGradients(float) override;

private:
    bool isInitialized;
    int neurons;
    string activationFunction;
    string layerName;

    Eigen::MatrixXf weights; 
    Eigen::MatrixXf gradients;
    Eigen::MatrixXf rawOutputs;
    Eigen::MatrixXf outputDeltas;
    Eigen::MatrixXf outputError;
    Eigen::VectorXf targets;
    Eigen::MatrixXf outputActivations;
    
    void initializeWeights(const int&, const int&);
    void backprop();


};

#endif // DENSE_HPP