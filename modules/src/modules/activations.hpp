#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP
#include <Eigen/Dense>


namespace activations{
    float sigmoid(float);
    float sigmoidDerivative(float);
    Eigen::MatrixXf softmax(Eigen::MatrixXf);
    float binaryCrossentropy(float, float); 
    float categoricalCrossentropy(float, float);
    Eigen::MatrixXf oneHotEncode(Eigen::VectorXf&, int);
    //Eigen::MatrixXf sparseCategoricalCrossentropy(Eigen::MatrixXf);
}

#endif

