#ifndef INPUT_HPP
#define INPUT_HPP
#include <Eigen/Dense>
#include "layer.hpp"
using namespace std;

class Input : public Layer{
    public:
        Input(Eigen::MatrixXf, string);
        string getLayerName() const override;
        Eigen::MatrixXf getOutputActivations() const override;
        Eigen::MatrixXf getDeltas() const override;
        Eigen::MatrixXf getWeights() const override;
        Eigen::MatrixXf getSigmoidDerivative() const override;
        Eigen::MatrixXf forward(const Eigen::MatrixXf&) override;
        void setDeltas(Eigen::MatrixXf&) override;
        void setGradients(Eigen::MatrixXf&) override;
        void applyGradients(float) override;

    private:
        string name;
        Eigen::MatrixXf inputs;
};


#endif