#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <typeinfo>
using namespace std;
using namespace Eigen;
using namespace cv;
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
    
}
