#ifndef DENSE_HPP
#define DENSE_HPP
#include <Eigen/Dense>
using namespace std;


class Dense {
public:
    Dense(int, string);


    Eigen::MatrixXf forward(const Eigen::MatrixXf&); //move to private later
    Eigen::MatrixXf getWeights();
    Eigen::MatrixXf setGradients();
    Eigen::MatrixXf setDeltas();
    Eigen::MatrixXf applyGradients();
    Eigen::MatrixXf getRawOutput();
    Eigen::MatrixXf getSigmoidDerivative();

private:
    bool isInitialized;
    int neurons;
    string activationFunction;

    Eigen::MatrixXf weights; 
    Eigen::MatrixXf gradients;
    Eigen::MatrixXf rawOutputs;
    Eigen::MatrixXf outputDeltas;
    Eigen::MatrixXf outputError;
    Eigen::VectorXf targets;
    
    void initializeWeights(const int&, const int&);
    void backprop();


};

#endif // DENSE_HPP