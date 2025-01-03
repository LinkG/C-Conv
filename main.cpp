#include "matrix.h"
#include <iostream>
#include "MNISTRead.H"
#include "CNN.h"
#include <string.h>
#include <time.h>

void display(float number[784]) {
    for(int i1 = 0; i1 < 28; i1++) {
        for(int j1 = 0; j1 < 28; j1++) {
            std::cout << (number[i1*28 + j1] == 1 ? 'o' : ' ') << ' ';
        }
        std::cout << '\n';
    }
}

void shuffleImagesAndLabels(bool** &images, char* &labels, int size, int row, int col) {
    int img_size = row * col;
    bool swap1[img_size];
    char swap2;
    int k = size / 2;
    for(int i = 0; i < k; i++) {
        int t = (rand() % k) + 1;
        memmove(swap1, images[i], sizeof(bool) * img_size);
        swap2 = labels[i];
        memmove(images[i], images[size - t], sizeof(bool) * img_size);
        labels[i] = labels[size - t];
        labels[size - t] = swap2;
        memmove(images[size - t], swap1, sizeof(bool) * img_size);
    }
}

int main(int argc, char** argv) {
    srand(time(0));
    //The options are: train mode, save file, config, path to data, num images, just name
    bool options[6] = {false, false, false, false, false, false};
    //Strings to store file names
    std::string save_name, config_name, data_path;

    //num of images
    int num_images = 10000, epochs = 0;
    float lr = 0.2;
    std::cout << argc << '\n';
    for(int i = 0; i < argc; ++i) {
        if(!strcasecmp(argv[i], "-m")) {
            if(!strcasecmp(argv[i + 1], "train")) {
                options[0] = true;
            } else if(strcasecmp(argv[i + 1], "infer")) {
                std::cout << "Invalid option for mode, valid options - train/infer\n";
                return 0;
            }
        } else if(!strcasecmp(argv[i], "-s")) {
            options[1] = true;
            save_name.append(argv[i + 1]);
        } else if(!strcasecmp(argv[i], "-c")) {
            options[2] = true;
            config_name.append(argv[i + 1]);
        } else if(!strcasecmp(argv[i], "-d")) {
            options[3] = true;
            data_path.append(argv[i + 1]);
            if(data_path.at(data_path.length() - 1) != '/') {
                data_path.append("/");
            }
        } else if(!strcasecmp(argv[i], "-n")) {
            options[4] = true;
            num_images = std::stoi(argv[i + 1]);
        } else if(!strcasecmp(argv[i], "-e")) {
            epochs = std::stoi(argv[i + 1]);
        } else if(!strcasecmp(argv[i], "-lr")) {
            lr = std::stof(argv[i + 1]);
        }
    }

    if(!options[3]) {
        data_path.append("Images/");
    }
    std::string images = data_path + "images-ubyte", labels = data_path + "labels-ubyte";

    MNISTData data(images, labels, num_images);

    ConvNet net;

    //Change this for different input sizes
    int picture_dims[2];
    picture_dims[0] = 28;
    picture_dims[1] = 28;
    bool** images_data = new bool*[num_images];
    char* labels_data = new char[num_images];
    for(int i = 0; i < num_images; ++i) {
        images_data[i] = new bool[picture_dims[0] * picture_dims[1]];
    }
    data.getImages(picture_dims[0], picture_dims[1], images_data);
    data.getLabels(labels_data);
    std::cout << "Loaded " << num_images << " image data!\n";
    bool dta = false;
    if(options[1]) {
        std::ifstream ftest(save_name, std::ios::binary);
        if(ftest.is_open()) {
            ftest.close();
            net.loadFromFile(save_name.c_str());
            dta = true;
        } else {
            options[1] = false;
            options[5] = true;
        }
    }
    if(options[2] && (options[5] || (!options[1] && !options[5]))) {
        net.loadConfig(config_name.c_str());
        dta = true;
    } 
    if(!dta) {
        std::cout << "Unable to get details about network, try using -s or -c to provide network details.\n";
        return 0;
    }

    if(net.inputdim[0] != picture_dims[0] || net.inputdim[1] != picture_dims[1]) {
        std::cout << "Network input and image size do not match.\n";
        return 0;
    }

    net.construct();
    std::cout << "Constrcuted required matrices.\n";
    if(!options[1]) {
        net.makeCNNRandom();
        std::cout << "Initialized new network successfully!\n";
    } else {
        std::cout << "Loaded network successfully!\n";
    }
    
    if(options[0]) {
        if(epochs == 0) {
            std::cout << "Enter epochs: ";
            std::cin >> epochs;
        }
        if(options[1] || options[5]) {
            std::cout << "Saving network to " << save_name << "\n";
        } else {
            std::cout << "Enter file name to save to: ";
            std::getline(std::cin, save_name);
        }
        for(; epochs > 0; --epochs) {
            net.descent(images_data, labels_data, num_images, lr, epochs - 1);
            shuffleImagesAndLabels(images_data, labels_data, num_images, picture_dims[0], picture_dims[1]);
            net.writeToFile(save_name.c_str());
            std::cout << "Completed one epoch and saved to file\n\n\n";
        }        
    } else {
        std::cout << "Running inference on " << num_images << " images.";
        int correct = 0;
        float max;
        int maxindex;
        Matrix img(net.inputdim[0], net.inputdim[1]);
        for(int j = 0; j < num_images; ++j) {
            max = 0, maxindex = 0;
            img.loadFromArray(images_data[j]);
            net.feedforward(img);
            for(int i = 0; i < net.layers[net.num_layers - 1]; ++i) {
                if(net.activations[net.num_layers - 1][i][0] > max) {
                    max = net.activations[net.num_layers - 1][i][0];
                    maxindex = i;
                }
            }
            if(maxindex == labels_data[j] - '0') {
                ++correct;
            }
            std::cout << "Number: " << labels_data[j] << " | Guessed: " << maxindex << " | Accuracy: " << (float)correct * 100 / j << '\n';
        }
    }

    return 0;
}