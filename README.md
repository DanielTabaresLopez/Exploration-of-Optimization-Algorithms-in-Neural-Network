# **Studying Stochastic Gradient Descent with MNIST**

This repository demonstrates the implementation of a simple neural network from scratch to study the **Stochastic Gradient Descent (SGD)** algorithm. The network is trained and evaluated using the **MNIST dataset**, a benchmark dataset for handwritten digit recognition. 

## **Objective**
The primary goal of this project is to provide a deeper understanding of how the SGD algorithm operates in training a neural network. By using the MNIST dataset as a concrete example, we explore:
- Forward propagation of data through the network.
- Backpropagation of gradients for optimizing the cost function.
- Iterative updates of weights and biases using mini-batches in SGD.

## **Overview**
This project implements a neural network with the following features:
- Configurable architecture: Fully connected layers specified by the user.
- Activation function: Sigmoid for non-linear transformations.
- Optimization: Stochastic Gradient Descent (SGD) with support for mini-batches.
- Dataset: Preprocessing of the MNIST dataset for training and evaluation.

The training process aims to minimize the cross-entropy loss, and the performance is evaluated based on classification accuracy on a test set.


## **How It Works**
1. **Dataset Preparation**:
   - The MNIST dataset is loaded using `keras.datasets`.
   - Training and test data are normalized and reshaped to match the input requirements of the network.
   - Labels are encoded in one-hot format for training and as integers for evaluation.

2. **Network Architecture**:
   - Fully connected layers with randomly initialized weights and biases.
   - Example configuration: `[784, 50, 10]` represents a network with:
     - 784 input neurons (for 28x28 pixel images).
     - 50 hidden layer neurons.
     - 10 output neurons (one for each digit class).

3. **Stochastic Gradient Descent**:
   - Training is performed in iterations.
   - Each iteration splits the training data into mini-batches.
   - Weights and biases are updated using gradients computed via backpropagation for each mini-batch.

4. **Evaluation**:
   - Accuracy is computed after each iteration to track the network's performance.


## **Results**
During training, the code outputs:
- The iteration number.
- Accuracy on the test set after each iteration.


## **Customization**
You can modify the following parameters in the script:

- Network Architecture: Specify the desired number of layers and neurons in the Network class initialization.
- Learning Rate: Adjust the rate of parameter updates during training.
- Mini-Batch Size: Set the size of data chunks processed in each SGD step.
- Iterations: Change the number of training iterations.
