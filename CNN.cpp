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

    for(int i = 0 ; i < num_layers; i++) {
        activations[i].createMatrix(layers[i], 1);
        gradients[i].createMatrix(layers[i], 1);
        pre_sigmoid[i].createMatrix(layers[i], 1);
        if(i == 0) {
            continue;
        }
        weights[i - 1].createMatrix(layers[i - 1], layers[i]);
        average_weights[i - 1].createMatrix(layers[i - 1], layers[i]);
        biases[i - 1].createMatrix(layers[i], 1);
        average_biases[i - 1].createMatrix(layers[i], 1);
    }
}

//Gives random values to all weights, biases and kernel using a normal distribuition
void ConvNet::makeCNNRandom() {
    kernel.doFunction(generateRandom);
    
    for(int i = 0; i < num_layers - 1; i++) {
        weights[i].doFunction(generateRandom);
        biases[i].doFunction(generateRandom);
    }
}

//Convolves and feedforward one image of type (Matrix)
void ConvNet::feedforward(Matrix &image) {
    //Hardcoding value
    Matrix convolved(24, 24);

    convolved = image.convolve(kernel);
    float* flat = convolved.flatten();
    
    activations[0].loadFromArray(flat);
    pre_sigmoid[0] = activations[0];

    pre_sigmoid[0].doFunction(sigmoidprime);
    activations[0].doFunction(sigmoid);

    delete[] flat;

    for(int i = 1; i < num_layers; i++) {
        activations[i] = (weights[i - 1].transpose() * activations[i - 1]) + biases[i - 1];
        pre_sigmoid[i] = activations[i];
        pre_sigmoid[i].doFunction(sigmoidprime);
        activations[i].doFunction(sigmoid);
    }
}

//Backpropagates the error in the neurons, requires the expected values and the original img matrix
//(float* correct_out, Matrix img)
//Make sure the expected array is as big as the final layer of activation else segfault
void ConvNet::backpropogate(Matrix &correct, Matrix &image) {
    gradients[num_layers - 1] = (activations[num_layers - 1] - correct) ^ pre_sigmoid[num_layers - 1];
    for(int i = num_layers - 2; i >= 0; i--) {
        gradients[i] = (weights[i] * gradients[i + 1]) ^ pre_sigmoid[i];
    }

    float* flat  = gradients[0].flatten();
    
    //Hardcoding values here again!
    Matrix gradient_matrix(24, 24);
    gradient_matrix.loadFromArray(flat);

    delete[] flat;

    kernel_gradient = kernel_gradient + image.convolve(gradient_matrix);
}

//Performs one itereation of gradient descent using the provided images, learning rate, and batch size
void ConvNet::descent(float** images, char* labels, int num_images, float learning_rate, int batch_size){
    int num_loops = num_images / batch_size;
    int sample_no;

    //Hardcoded values
    Matrix img(28, 28);
    Matrix correct(10, 1);
    correct = correct * 0;

    for(int i = 0; i < num_loops; i++) {
        kernel_gradient = kernel_gradient * 0;
        for(int j = 0; j < num_layers - 1; j++) {
            average_biases[j] = average_biases[j] * 0;
            average_weights[j] = average_weights[j] * 0;
        }
        for(int j = 0; j < batch_size; j++) {
            sample_no = (i * batch_size) + j;
            img.loadFromArray(images[sample_no]);
            correct[labels[sample_no] - '0'][0] = 1;
            
            feedforward(img);
            backpropogate(correct, img);

            correct[labels[sample_no] - '0'][0] = 0;

            for(int k = 0; k < num_layers - 1; k++) {
                average_biases[k] = average_biases[k] + gradients[k + 1];
                average_weights[k] = average_weights[k] + (activations[k] * (gradients[k + 1].transpose()));
            }
        }
        kernel = kernel - (kernel_gradient * (learning_rate / batch_size));
        for(int j = 0; j < num_layers - 1; j++) {
            weights[j] = weights[j] - (average_weights[j] * (learning_rate / batch_size));
            biases[j] = biases[j] - (average_biases[j] * (learning_rate / batch_size));
        }
    }
}
