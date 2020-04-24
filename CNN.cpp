#include "CNN.h"

std::default_random_engine generator;
std::normal_distribution<float> distribuition(0.0f, 1.0f);
float generateRandom(float x) {
    return distribuition(generator);
}

float sigmoid(float x){
    float xi = 1.0 / (1 + exp(-x));
    if(xi == 1) {
        return float(0.999);
    } else if(xi == 0) {
        return float(0.001);
    } else {
        return xi;
    }
}

ConvNet::ConvNet():kernel(5, 5){
    layers[0] = Matrix::getColConvolve(28, 5) * Matrix::getRowConvolve(28, 5);
    
    //Hardcoding the values, change to softcoded when needed
    img.createMatrix(28, 28);
    convolved.createMatrix(24, 24);

    for(int i = 0 ; i < num_layers; i++) {
        activations[i].createMatrix(layers[i], 1);
        if(i == 0) {
            continue;
        }
        weights[i - 1].createMatrix(layers[i - 1], layers[i]);
        biases[i - 1].createMatrix(layers[i], 1);
    }
}

void ConvNet::makeCNNRandom() {
    for(int i = 0; i < kernel.dimension[0]; i ++) {
        for(int j = 0; j < kernel.dimension[1]; j++) {
            kernel[i][j] = distribuition(generator);
        }
    }
    
    for(int i = 0; i < num_layers - 1; i++) {
        weights[i].doFunction(generateRandom);
        biases[i].doFunction(generateRandom);
    }
}

void ConvNet::feedforward(float* image) {
    img.loadFromArray(image);
    convolved = img.convolve(kernel);
    float* flat = convolved.flatten();
    activations[0].loadFromArray(flat);
    activations[0].doFunction(sigmoid);
    delete[] flat;

    for(int i = 1; i < num_layers; i++) {
        activations[i] = weights[i - 1].transpose() * activations[i - 1] + biases[i - 1];
        activations[i].doFunction(sigmoid);
    }
}
