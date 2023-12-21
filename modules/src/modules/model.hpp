#ifndef MODEL_HPP
#define MODEL_HPP
#include "dense.hpp"
#include <Eigen/Dense>
using namespace std;

class Model{
    public:
        Model(string, int);
        void addLayer(Dense);
        void fit(Eigen::MatrixXf&, Eigen::VectorXf&);

    private:
        int batchSize;
        string lossFunction;
        vector<Dense> layers;
        Eigen::VectorXf labels;

        Eigen::MatrixXf forwardPass(Eigen::MatrixXf&);
        void backprop();

};



#endif // DENSE_HPP