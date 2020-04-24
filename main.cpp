#include "matrix.h"
#include <iostream>
#include "MNISTRead.H"
#include "CNN.h"

int main() {
    int num_img = 1, size;
    MNISTData data("Images/images-ubyte", "Images/labels-ubyte", num_img);
    float** images = data.getImages(size);
    char* labels = data.getLabels();
    
    ConvNet net;

    net.makeCNNRandom();
    net.feedforward(images[0]);
    for(int i = 0; i < net.activations[2].dimension[0]; i++) {
        std::cout << net.activations[2][i][0] << ", ";
    }
    std::cout << "\nYAY" << "\n";
    return 0;
}