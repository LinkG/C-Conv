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

    std::cout << "RUNNING\n";

    ConvNet net;
    net.loadConfig("network_prop.txt");
    net.construct();
    net.makeCNNRandom();
    for(int i = 0; i < 5; i++) {
        shuffleImagesAndLabels(images, labels, num_img);
        net.descent(images, labels, num_img, 1.2);
        std::cout << i + 1 << "\n";
    }


    //Inference
    int corr = 0;
    int t = net.num_layers - 1;
    for(int i = 0; i < num_img; i++) {
        img.loadFromArray(images[i]);
        net.feedforward(img);
        int max = 0;
        float val = 0;
        for(int j = 0; j < net.activations[t].dimension[0]; j++) {
            if(net.activations[t][j][0] > val) {
                val = net.activations[t][j][0];
                max = j;
            }
        }
        if(max == (labels[i] - '0')) {
            corr++;
        }
        std::cout << "\n" << (i + 1) << ") " << labels[i] << " : " << (corr * 100.0f)/(i + 1) << "\n";
    }

    int ch = 0;
    std::cout << "\n\nWould you like to save(y/n):\n";
    std::cin >> ch;
    if(ch == 'y') {
        net.writeToFile("Networksave.dat");
        std::cout << "\nWrote to \"Networksave.dat\"\n";
    }
    return 0;
}