#include <iostream>
#include <modules/dense.hpp>
#include <modules/model.hpp>
#include <modules/input.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <modules/activations.hpp>
#include <memory>
#include <dirent.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
using namespace std;
typedef unsigned char uchar;


void shuffleData(Eigen::MatrixXf& images, Eigen::VectorXf& labels);
std::pair<Eigen::MatrixXf, Eigen::VectorXf> read_mnist_cv2(const char* image_filename, const char* label_filename);


int main(){
    //dataset paths
    string base_dir = "/Users/ulrikisdahl/Desktop/machine learning/mnist/";
    string img_path = base_dir + "train-images.idx3-ubyte";
    string label_path = base_dir + "train-labels.idx1-ubyte";

    //load training data
    std::pair<Eigen::MatrixXf, Eigen::VectorXf> results = read_mnist_cv2(img_path.c_str(), label_path.c_str());
    Eigen::MatrixXf images = results.first;
    Eigen::VectorXf labels = results.second;

    //load test data
    string img_path_test = base_dir + "t10k-images.idx3-ubyte";
    string label_path_test = base_dir + "t10k-labels.idx1-ubyte";
    std::pair<Eigen::MatrixXf, Eigen::VectorXf> testData = read_mnist_cv2(img_path.c_str(), label_path.c_str());
    Eigen::MatrixXf testImages = testData.first;
    Eigen::VectorXf testLabels = testData.second;
    Eigen::MatrixXf testOneHotEcondedLabels = activations::oneHotEncode(testLabels, 10);

    //preprocess dataset
    shuffleData(images, labels);
    Eigen::MatrixXf oneHotEncodedLabels = activations::oneHotEncode(labels, 10);


    //initializing the model architecture
    Model model = Model();

    model.addLayer(std::make_unique<Input>(images, "inputLayer")); 

    model.addLayer(std::make_unique<Dense>(64, "sigmoid", "hidden1"));

    model.addLayer(std::make_unique<Dense>(128, "sigmoid", "hidden2"));

    model.addLayer(std::make_unique<Dense>(10, "softmax", "output")); //output layer for multiclass classification

    model.fit(images, oneHotEncodedLabels, 8, 0.0001f);

    //evaluate the model's performance on test dataset
    model.evaluate(testImages, testOneHotEcondedLabels, true);

    return 0;
}


/// @brief Swaps the endianess of a 32-bit unsigned integer.
/// @param val The value to swap.
uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}


/**
 * @brief Reads the MNIST dataset from binary files and loads it into Eigen matrices.
 *
 * @param image_filename The path to the MNIST image file in binary format.
 * @param label_filename The path to the MNIST label file in binary format.
 * @return A pair of Eigen::MatrixXf and Eigen::VectorXf containing the images and labels respectively.
 *         Returns an empty pair if there's an error in reading files or if the files' magic numbers
 *         don't match the expected MNIST format.
 *
 * Reference:
 * The reading logic of the MNIST data set is based on a solution provided in a Stack Overflow post:
 * https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c
 */
std::pair<Eigen::MatrixXf, Eigen::VectorXf> read_mnist_cv2(const char* image_filename, const char* label_filename) {
    // Open files
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

    //Read meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if(magic != 2051){
        cout<<"Incorrect image file magic: "<<magic<<endl;
        return std::pair<Eigen::MatrixXf, Eigen::VectorXf>();
    }

    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if(magic != 2049){
        cout<<"Incorrect image file magic: "<<magic<<endl;
        return std::pair<Eigen::MatrixXf, Eigen::VectorXf>();
    }

    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    if(num_items != num_labels){
        cout<<"image file nums should equal to label num"<<endl;
        return std::pair<Eigen::MatrixXf, Eigen::VectorXf>();
    }

    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    cout<<"image and label num is: "<<num_items<<endl;
    cout<<"image rows: "<<rows<<", cols: "<<cols<<endl;


    //Read the rest of the data into memory
    // Define Eigen matrix and vector to hold all the images and labels
    Eigen::MatrixXf images(num_items, rows * cols);
    Eigen::VectorXf labels(num_items);

    unsigned char label;
    unsigned char* pixels = new unsigned char[rows * cols];

    for (int item_id = 0; item_id < num_items; ++item_id) {
        // Read image pixel
        image_file.read(reinterpret_cast<char*>(pixels), rows * cols);
        // Read label
        label_file.read(reinterpret_cast<char*>(&label), 1);

        // Convert pixel data to Eigen::VectorXf and store in matrix
        for (int i = 0; i < rows * cols; ++i) {
            images(item_id, i) = static_cast<float>(pixels[i]) / 255.0f;
        }

        // Store label
        labels[item_id] = static_cast<float>(label);
    }

    delete[] pixels;

    return std::make_pair(images, labels);
}


/// @brief shuffles data
/// @param images Eigen::MatrixXf reference of images
/// @param labels Eigen::MatrixXf reference of labels
void shuffleData(Eigen::MatrixXf& images, Eigen::VectorXf& labels) {
    // Make sure images and labels have the same size
    if (images.rows() != labels.size()) {
        throw std::runtime_error("Images and labels must have the same number of elements");
    }

    // Create a vector of indices
    std::vector<size_t> indices(images.rows());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Shuffle the indices
    std::random_device rd; 
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Create temporary containers for the shuffled data
    Eigen::MatrixXf shuffledImages(images.rows(), images.cols());
    Eigen::VectorXf shuffledLabels(labels.size());

    // Reorder the data according to the shuffled indices
    for (size_t i = 0; i < indices.size(); ++i) {
        shuffledImages.row(i) = images.row(indices[i]);
        shuffledLabels(i) = labels(indices[i]);
    }

    // Swap the original data with the shuffled data
    images.swap(shuffledImages);
    labels.swap(shuffledLabels);
}
