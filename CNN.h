#ifndef CNN
#define CNN

#include "matrix.h"
#include <random>

class ConvNet {
public:
    int num_layers = 3;
    int layers[3] = {0, 50, 10};
    Matrix weights[2];
    Matrix biases[2];
    Matrix activations[3];
    Matrix kernel, img, convolved;
    ConvNet();
    void makeCNNRandom();
    void feedforward(float* image);
};

#endif