#include "layer.hpp"
#include "input.hpp"
#include <iostream>
#include <Eigen/Dense>

using namespace std;


Input::Input(Eigen::MatrixXf input){
    this->inputs = input;
}

Eigen::MatrixXf Input::getOutputActivations(){
    return this->inputs;
}
