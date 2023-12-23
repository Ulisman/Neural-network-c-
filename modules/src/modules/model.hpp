#ifndef MODEL_HPP
#define MODEL_HPP
#include "dense.hpp"
#include <Eigen/Dense>
#include <memory>
using namespace std;

class Model{
    public:
        Model(string, int);
        void addLayer(std::unique_ptr<Layer>);
        void fit(Eigen::MatrixXf&, Eigen::VectorXf&, float);

    private:
        int batchSize;
        string lossFunction;
        vector<std::unique_ptr<Layer>> layers;
        Eigen::VectorXf labels;

        Eigen::MatrixXf forwardPass(Eigen::MatrixXf&);
        void backprop();

};



#endif // DENSE_HPP