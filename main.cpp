#include "matrix.h"
#include <iostream>
#include "MNISTRead.H"
#include "CNN.h"
#include <string.h>

void shuffleImagesAndLabels(float** &images, char* &labels, int size) {
    float swap1[784];
    char swap2;
    int k = size / 2;
    for(int i = 0; i < k; i++) {
        int t = (rand() % k) + 1;
        memmove(swap1, images[i], sizeof(float) * 784);
        swap2 = labels[i];
        memmove(images[i], images[size - t], sizeof(float) * 784);
        labels[i] = labels[size - t];
        labels[size - t] = swap2;
        memmove(images[size - t], swap1, sizeof(float) * 784);
    }
}

int main() {
    srand(time(0));
    int num_img = 10000, size;
    MNISTData data("Images/images-ubyte", "Images/labels-ubyte", num_img);
    float** images = data.getImages(size);
    char* labels = data.getLabels();
    Matrix img(28, 28);
    img.loadFromArray(images[0]);
    char fst = labels[0];

    ConvNet net;
    net.makeCNNRandom();
    for(int i = 0; i < 20; i++) {
        shuffleImagesAndLabels(images, labels, num_img);
        net.descent(images, labels, num_img, 1);
        std::cout << i + 1 << "\n";
        net.feedforward(img);
        for(int i = 0; i < net.activations[2].dimension[0]; i++) {
            std::cout << net.activations[2][i][0] << ", ";
        }
        std::cout << fst << "\n";
    }
    std::cout << "\nYAY" << "\n";
    return 0;
}