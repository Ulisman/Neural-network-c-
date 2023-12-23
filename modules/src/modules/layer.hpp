#ifndef LAYER_HPP
#define LAYER_HPP
#include <Eigen/Dense>
using namespace std;

class Layer{
    public:
        virtual ~Layer() = default;
        virtual string getLayerName() const = 0; //dummy definition for the virtual function
        virtual Eigen::MatrixXf getOutputActivations() const = 0; 
        virtual Eigen::MatrixXf getDeltas() const = 0;
        virtual Eigen::MatrixXf getWeights() const = 0;
        virtual Eigen::MatrixXf getSigmoidDerivative() const = 0;
        virtual Eigen::MatrixXf forward(const Eigen::MatrixXf&) = 0; //not const because the method changes the state of the object
        virtual void setDeltas(Eigen::MatrixXf&) = 0;
        virtual void setGradients(Eigen::MatrixXf&) = 0;
        virtual void applyGradients(float) = 0;

    private:
};


#endif