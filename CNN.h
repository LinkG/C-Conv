#ifndef CNN
#define CNN

#include "matrix.h"
#include <random>

class ConvNet {
public:
    //These are hardcoded, switch to softcoded later
    int num_layers = 4;
    int layers[4] = {0, 100, 30, 10};

    //Same here
    Matrix weights[3], biases[3], activations[4], gradients[4], pre_sigmoid[4],
            average_weights[3], average_biases[3];
    Matrix kernel, kernel_gradient;

    //Constructor, constructs kernel and create the rest of the matrices, also
    //initializes the first layer of the network depending on the convolved size
    ConvNet();

    //Gives random values to all weights, biases and kernel using a normal distribuition
    void makeCNNRandom();

    //Convolves and feedforward one image of type (Matrix)
    void feedforward(Matrix &img);

    //Backpropagates the error in the neurons, requires the expected values and the original img matrix
    //(float* correct_out, Matrix img)
    //Make sure the expected array is as big as the final layer of activation else segfault
    void backpropogate(Matrix &expected, Matrix &img);

    //Performs one itereation of gradient descent using the provided images, learning rate, and batch size
    void descent(float** images, char* labels, int num_images, float learning_rate, int batch_size=8);
};

#endif