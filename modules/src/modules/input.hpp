#ifndef LAYER_HPP
#define LAYER_HPP
#include <Eigen/Dense>
#include "layer.hpp"
using namespace std;

class Input : public Layer{
    public:
        Input(Eigen::MatrixXf);
        Eigen::MatrixXf getOutputActivations();
        //void setInputBatch(Eigen::MatrixXf);

    private:
        Eigen::MatrixXf inputs;
};


#endif