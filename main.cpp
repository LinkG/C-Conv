#include "matrix.h"
#include <iostream>
#include "MNISTRead.H"
#include "CNN.h"
#include <string.h>

void display(float number[784]) {
    for(int i1 = 0; i1 < 28; i1++) {
        for(int j1 = 0; j1 < 28; j1++) {
            std::cout << (number[i1*28 + j1] == 1 ? 'o' : ' ') << ' ';
        }
        std::cout << '\n';
    }
}

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
    for(int i = 0; i < 50; i++) {
        shuffleImagesAndLabels(images, labels, num_img);
        net.descent(images, labels, num_img, 1.2);
        std::cout << i + 1 << "\n";
    }

    int corr = 0;
    for(int i = 0; i < num_img; i++) {
        img.loadFromArray(images[i]);
        net.feedforward(img);
        int max = 0;
        float val = 0;
        for(int j = 0; j < net.activations[3].dimension[0]; j++) {
            std::cout << net.activations[3][j][0] << ",";
            if(net.activations[3][j][0] > val) {
                val = net.activations[3][j][0];
                max = j;
            }
        }
        if(max == (labels[i] - '0')) {
            corr++;
        }
        std::cout << "\n" << labels[i] << " : " << (corr * 100.0f)/(i + 1) << "\n";
    }
    std::cout << "\nYAY" << "\n";
    return 0;
}