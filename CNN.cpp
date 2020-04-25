#include "CNN.h"

//Normal distribuition random number generator
std::default_random_engine generator;
std::normal_distribution<float> distribuition(0.0f, 1.0f);
float generateRandom(float x) {
    return distribuition(generator);
}


//Sigmoid function used as a logistic curve
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

//Derivative of sigmoid evauated at x
float sigmoidprime(float x) {
    float fl = sigmoid(x);
    return fl * (1 - fl);
}

//Constructor, constructs kernel and create the rest of the matrices, also
//initializes the first layer of the network depending on the convolved size
ConvNet::ConvNet():kernel(5, 5), kernel_gradient(5, 5){
    layers[0] = Matrix::getColConvolve(28, 5) * Matrix::getRowConvolve(28, 5);
    
    //Hardcoding the values, change to softcoded when needed
    img.createMatrix(28, 28);
    convolved.createMatrix(24, 24);

    for(int i = 0 ; i < num_layers; i++) {
        activations[i].createMatrix(layers[i], 1);
        gradients[i].createMatrix(layers[i], 1);
        pre_sigmoid[i].createMatrix(layers[i], 1);
        if(i == 0) {
            continue;
        }
        weights[i - 1].createMatrix(layers[i - 1], layers[i]);
        biases[i - 1].createMatrix(layers[i], 1);
    }
}

//Gives random values to all weights, biases and kernel using a normal distribuition
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

//Convolves and feedforward one image of type (flaot*)
void ConvNet::feedforward(float* image) {
    img.loadFromArray(image);
    convolved = img.convolve(kernel);
    float* flat = convolved.flatten();
    activations[0].loadFromArray(flat);
    activations[0].doFunction(sigmoid);
    delete[] flat;

    for(int i = 1; i < num_layers; i++) {
<<<<<<< HEAD
<<<<<<< HEAD
        activations[i] = (weights[i - 1].transpose() * activations[i - 1]) + biases[i - 1];
        pre_sigmoid[i] = activations[i];
        pre_sigmoid[i].doFunction(sigmoidprime);
=======
>>>>>>> a2f5001a9262e7936f84c502a505b967db71be29
=======
        activations[i] = (weights[i - 1].transpose() * activations[i - 1]) + biases[i - 1];
>>>>>>> a2f5001a9262e7936f84c502a505b967db71be29
        activations[i].doFunction(sigmoid);
    }
}

//Backpropagates the error in the neurons, requires the expected values and the original img matrix
//(float* correct_out, Matrix img)
//Make sure the expected array is as big as the final layer of activation else segfault
void ConvNet::backpropogate(float* expected, Matrix &img) {
    Matrix correct(activations[num_layers - 1].dimension[0], 1);
    correct.loadFromArray(expected);

    gradients[num_layers - 1] = (activations[num_layers - 1] - correct) ^ pre_sigmoid[num_layers - 1];
    for(int i = num_layers - 2; i >= 0; i--) {
        gradients[i] = (weights[i] * gradients[i + 1]) ^ pre_sigmoid[i];
    }

    float* flat  = gradients[0].flatten();
    
    //Hardcoding values here again!
    Matrix gradient_matrix(24, 24);
    gradient_matrix.loadFromArray(flat);

    delete[] flat;

    kernel_gradient = img.convolve(gradient_matrix);
}

//Performs one itereation of gradient descent using the provided images, learning rate, and batch size
void ConvNet::descent(float** images, char* labels, int num_images, float learning_rate, int batch_size){

}
