#ifndef MODEL_HPP
#define MODEL_HPP
#include "dense.hpp"
#include <Eigen/Dense>
#include <memory>
using namespace std;

class Model{
    public:
        Model();
        void addLayer(std::unique_ptr<Layer>);
        void fit(Eigen::MatrixXf&, Eigen::MatrixXf&, int, float);
        void evaluate(Eigen::MatrixXf, Eigen::MatrixXf&, bool);
        Eigen::MatrixXf predict(Eigen::MatrixXf&);

    private:
        int batchSize;
        string lossFunction;
        vector<std::unique_ptr<Layer>> layers;
        Eigen::VectorXf labels;

        Eigen::MatrixXf forwardPass(Eigen::MatrixXf&);
        void backprop(Eigen::MatrixXf&, Eigen::MatrixXf&, float&);

};



#endif // DENSE_HPP