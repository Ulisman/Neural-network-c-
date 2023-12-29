# C++ Neural Networks

A neural network library for creating simple and fast models in c++

## Why?

C++ goes brr...

## Example
[Simple model with two hidden layers](inference/main.cpp)
<br>
- Above is an example of a model with two hidden layers used for multiclass classification on the mnist dataset

## Syntax
<br>
Initialize a model:

```
Model model = Model();
```
<br>
Add input layer first (cruical): 

```
model.addLayer(std::unique_ptr<Dense>(input_data, "layer_name"))
```
<br>
Add linear layer (dense) with activation function:

```
model.addLayer(std::unique_ptr<Dense>(num_neurons, "activation", "layer_name"))
```
- NB! The layers passed into the method must be smart pointers of the class instance

<br>
Train the model:

```
model.fit(input_data, labels, epochs, learning_rate)
```
- NB! `labels` must be one-hot encoded

##
