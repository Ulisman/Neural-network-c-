#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <typeinfo>
using namespace std;
using namespace Eigen;
using namespace cv;
float returnMinus(float);
float sigmoid(float);
float binaryError(float, float);

int main(){ 
    //Standard matrix
    Matrix <float, 3, 3> matrixA;
    matrixA.setZero();
    cout << matrixA <<endl;
    cout << matrixA(0,1) <<endl;
    cout << matrixA.row(0) <<endl;
    cout << matrixA.col(0) <<endl;

    // Dynamic matrix 
    Matrix <float, Dynamic, Dynamic> matrixB;

    //Matrix operations
    Matrix <float, 3, 3> B;
    B << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;
    cout << B <<endl;

    MatrixXf C(3,3);
    C << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;
    cout << C << endl;
    //Matrix addition
    cout << B + C << endl;

    //multiplication
    cout << B * C << endl;

    //transpose
    MatrixXf D(3,3);
    D = B.transpose();
    cout << D << endl;

    //Set a matrix to be its transpose - C = C.transpose() is wrong!
    C.transposeInPlace();  
    cout << C << endl;

    //load image
    cv::Mat img = cv::imread("/Users/ulrikisdahl/Desktop/machine learning/mnist/trainingSet/trainingSet/0/img_965.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(8,8));

    //covnert image to Eigen
    Matrix<float, 8, 8> eigenImage;
    for (int i = 0; i < resized.rows; i++){
        for (size_t j = 0; j < resized.cols; j++){
            eigenImage(i, j) = resized.at<uchar>(i,j);
        }
    }

    cout << eigenImage * 0.99 << endl;

    Map<VectorXf> v(eigenImage.data(), eigenImage.size());
    cout << v << endl;
    cout << typeid(v).name() << endl;
    

    Matrix<float, 3, 6> nice;
    nice << 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 
    cout << D * nice << endl;

    //vectorize nice matrix with using the returnMinus function
    //std::function<float(float)> func = returnMinus;
    Matrix<float, 3, 6> result2 = nice.unaryExpr(std::function<float(float)>(returnMinus));

    cout << result2 << endl;

    cout << "test" << endl;

    VectorXf test(3);
    test << 3, 3, 3;

    VectorXf test2(3);
    test2 << 4, 5, 6;

    VectorXf resultTest(3);
    resultTest = test2 - test;
    cout << resultTest << endl;

    std::function<float(float)> func = sigmoid;
    VectorXf sigmoidVec = resultTest.unaryExpr(func);

    cout << sigmoidVec << endl;


    VectorXf outputs(3);
    outputs << 0.1, 0.2, 0.3;
    VectorXf labels(3);
    labels << 0, 1, 0;

    std::function<float(float, float)> func2 = binaryError;
    VectorXf error = outputs.binaryExpr(labels, func2);


    //dot porduct
    MatrixXf threeByFour(3, 4);
    threeByFour << 1, 1, 1, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1;

    MatrixXf dotProd = labels.transpose() * threeByFour; // (3, 1).T * (3, 4) = (1, 4)
    cout << dotProd << endl;

    cout << labels.rows() << endl;
    cout << labels.transpose().rows() << endl;
    cout << labels.rows() << endl;

}

float returnMinus(float elem){
    return elem - 4;
};

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float binaryError(float x, float y){
    return -y * std::log(x) - (1 - y) * std::log(1 - x);
}

// Eigen::VectorXf sigmoid(const Eigen::VectorXf& x) {
//     return x.unaryExpr([](float v) { return 1.0f / (1.0f + std::exp(-v)); });
// }


