#include "CNN.h"

//reads int from comma seperated string
int getInt(std::string &str) {
    int result = atoi(&str.substr(0, str.find(','))[0]);
    str.erase(0, str.find(',') + 1);
    return result;
}

//reads float from comma seperated string
float getFloat(std::string &str) {
    float result = atof(&str.substr(0, str.find(','))[0]);
    str.erase(0, str.find(',') + 1);
    return result;
}

//Splits string at new line
std::string split(std::string &str) {
    std::string result = str.substr(0, str.find('\n'));
    str.erase(0, str.find('\n') + 1);
    return result;
}

int numLine(std::string &file) {
    int n = 0;
    for(int i = 0; i < file.length(); i++) {
        if(file[i] == '\n') {
            n++;
        }
    }
    return n;
}

//Reads property from file
std::string readProperty(std::string file, const std::string property) {
    std::string value;
    std::string new_str(file);

    int num_lines = numLine(new_str);
    for(int i = 0; i < num_lines + 1; i++) {
        std::string line = split(new_str);
        if(line.find(property) == 0) {
            value = line.substr(property.length() + 1);
            break;
        }
    }

    value.erase(std::remove_if(value.begin(), value.end(), isspace), value.end());
    return value;
}


//Reads an array from file
int* readArrayInt(std::ifstream &file, int size) {
    int* ret = new int[size];
    int temp = 0;
    for(int i = 0; i < size; i++) {
        file.read((char*) &temp, sizeof(int));
        ret[i] = temp;
    }
    return ret;
}

float* readArrayFloat(std::ifstream &file, int size) {
    float* ret = new float[size];
    float temp = 0;
    for(int i = 0; i < size; i++) {
        file.read((char*) &temp, sizeof(float));
        ret[i] = temp;
    }
    return ret;
}

//Writes an array to a file
void writeArray(std::ofstream &file, float* arr, int size) {
    for(int i = 0;i < size; i++) {
        file.write((char*) &(arr[i]), sizeof(float));
    }
}

void writeArray(std::ofstream &file, int* arr, int size) {
    for(int i = 0;i < size; i++) {
        file.write((char*) &(arr[i]), sizeof(int));
    }
}

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

ConvNet::ConvNet() {
    isConstructed = false;
    layers = nullptr;
    num_layers = 0;
}

//Constructs kernel and create the rest of the matrices, also
//initializes the first layer of the network depending on the convolved size
void ConvNet::construct() {
    average_weights = new Matrix[num_layers - 1];
    average_biases = new Matrix[num_layers - 1];
    activations = new Matrix[num_layers];
    gradients = new Matrix[num_layers];
    pre_sigmoid = new Matrix[num_layers];

    kernel_filter_gradient = new Matrix[num_kernels];
    kernel_network_average_weights = new Matrix*[num_kernels];
    kernel_network_average_biases = new Matrix*[num_kernels];
    kernel_network_activations = new Matrix*[num_kernels];
    kernel_gradient = new Matrix*[num_kernels];
    kernel_network_pre_sigmoid = new Matrix*[num_kernels];

    for(int i = 0; i < num_kernels; ++i) {
        kernel_network_average_weights[i] = new Matrix[kernel_num_layers[i] - 1];
        kernel_network_average_biases[i] = new Matrix[kernel_num_layers[i] - 1];
        kernel_network_activations[i] = new Matrix[kernel_num_layers[i]];
        kernel_gradient[i] = new Matrix[kernel_num_layers[i]];
        kernel_network_pre_sigmoid[i] = new Matrix[kernel_num_layers[i]];
        kernel_filter_gradient[i].createMatrix(kernel[i].dimension[0], kernel[i].dimension[1]);

        for(int j = 0; j < kernel_num_layers[i]; ++j) {
            kernel_network_activations[i][j].createMatrix(kernel_network_layers[i][j], 1);
            kernel_gradient[i][j].createMatrix(kernel_network_layers[i][j], 1);
            kernel_network_pre_sigmoid[i][j].createMatrix(kernel_network_layers[i][j], 1);
            if(j == 0) {
                continue;
            }
            kernel_network_average_biases[i][j - 1].createMatrix(kernel_network_layers[i][j], 1);
            kernel_network_average_weights[i][j - 1].createMatrix(kernel_network_layers[i][j-1], kernel_network_layers[i][j]);
        }
    }

    for(int i = 0 ; i < num_layers; i++) {
        activations[i].createMatrix(layers[i], 1);
        gradients[i].createMatrix(layers[i], 1);
        pre_sigmoid[i].createMatrix(layers[i], 1);
        if(i == 0) {
            continue;
        }
        average_weights[i - 1].createMatrix(layers[i - 1], layers[i]);
        average_biases[i - 1].createMatrix(layers[i], 1);
    }

    isConstructed = true;
}

//Gives random values to all weights, biases and kernel using a normal distribuition
void ConvNet::makeCNNRandom() {
    if(!isConstructed) {
        throw std::invalid_argument("Unconstructed network");
        return;
    }

    for(int i = 0; i < num_kernels; ++i) {
        kernel[i].doFunction(generateRandom);
        for(int j = 0; j < kernel_num_layers[i]; ++j) {
            kernel_network_weights[i][j].doFunction(generateRandom);
            kernel_network_biases[i][j].doFunction(generateRandom);
        }
    }
    
    for(int i = 0; i < num_layers - 1; i++) {
        weights[i].doFunction(generateRandom);
        biases[i].doFunction(generateRandom);
    }
}

//Convolves and feedforward one image of type (Matrix)
void ConvNet::feedforward(Matrix &image) {
    if(!isConstructed) {
        throw std::invalid_argument("Unconstructed network");
        return;
    }

    for(int i = 0; i < num_kernels; ++i) {
        Matrix convolved(Matrix::getRowConvolve(inputdim[0], kernel[i].dimension[0]), Matrix::getColConvolve(inputdim[1], kernel[i].dimension[1]));
        convolved = image.convolve(kernel[i]);
        float* flat = convolved.flatten();
        kernel_network_activations[i][0].loadFromArray(flat);
        kernel_network_pre_sigmoid[i][0] = kernel_network_activations[i][0];

        kernel_network_pre_sigmoid[i][0].doFunction(sigmoidprime);
        kernel_network_activations[i][0].doFunction(sigmoid);

        delete[] flat;

        for(int j = 1; j < kernel_num_layers[i]; ++j) {
            kernel_network_activations[i][j] = (kernel_network_weights[i][j-1].transpose() * kernel_network_activations[i][j-1]) + kernel_network_biases[i][j-1];
            kernel_network_pre_sigmoid[i][j] = kernel_network_activations[i][j];

            kernel_network_pre_sigmoid[i][j].doFunction(sigmoidprime);
            kernel_network_activations[i][j].doFunction(sigmoid);
        }
    }

    float* flat = new float[layers[0]];
    int pos = 0;
    for(int i = 0; i < num_kernels; ++i) {
        float* temp_flat = kernel_network_activations[i][kernel_num_layers[i] - 1].flatten();
        int len = kernel_network_activations[i][kernel_num_layers[i] - 1].dimension[0];

        std::copy(temp_flat, temp_flat + len, flat + pos);

        pos += len;

        delete[] temp_flat;
    }

    //No max pooling rn
    //Matrix max_convolved = maxPool(convolved, 2, 2, 1);
    // float* flat = max_convolved.flatten();
    
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

//Performs max pooling operation, requires dimensions(row, col) and stride
Matrix ConvNet::maxPool(Matrix &img, int row, int col, int stride) {
    int d[2] = {((img.dimension[0] - row) / stride) + 1, ((img.dimension[1] - col) / stride) + 1};
    float t_arrr[d[0] * d[1]];
    int p = 0;
    for(int i = 0; i < img.dimension[0] - row; i += stride) {
        for(int j = 0; j < img.dimension[1] - col; j += stride) {
            float max_num = img[i][j];
            for(int k = i; k < i + row; ++k) {
                for(int l = j; l < j + col; ++l) {
                    if(img[k][l] > max_num) {
                        max_num = img[k][l];
                    }
                }
            }
            t_arrr[p++] = max_num;
        }
    }
    Matrix ret(d[0], d[1]);
    ret.loadFromArray(t_arrr);
    return ret;
}

//Backpropagates the error in the neurons, requires the expected values and the original img matrix
//(float* correct_out, Matrix img)
//Make sure the expected array is as big as the final layer of activation else segfault
void ConvNet::backpropogate(Matrix &correct, Matrix &image) {
    if(!isConstructed) {
        throw std::invalid_argument("Unconstructed network");
        return;
    }

    gradients[num_layers - 1] = (activations[num_layers - 1] - correct) ^ pre_sigmoid[num_layers - 1];
    for(int i = num_layers - 2; i >= 0; i--) {
        gradients[i] = (weights[i] * gradients[i + 1]) ^ pre_sigmoid[i];
    }

    float* flat  = gradients[0].flatten();
    int pos = 0;
    for(int i = 0; i < num_kernels; ++i) {
        int len = kernel_gradient[i][kernel_num_layers[i] - 1].dimension[0];
        float* temp_flat = new float[len];

        std::copy(flat + pos, flat + len, temp_flat);
        kernel_gradient[i][kernel_num_layers[i] - 1].loadFromArray(temp_flat);
        pos += len;

        delete[] temp_flat;
    }
    delete[] flat;

    for(int i = 0; i < num_kernels; ++i) {
        for(int j = kernel_num_layers[i] - 2; j >= 0; --j) {
            kernel_gradient[i][j] = (kernel_network_weights[i][j] * kernel_gradient[i][j + 1]) ^ kernel_network_pre_sigmoid[i][j];
        }

        float* flat = kernel_gradient[i][0].flatten();
        Matrix gradient_matrix(Matrix::getRowConvolve(inputdim[0], kernel[i].dimension[0]), Matrix::getColConvolve(inputdim[1], kernel[i].dimension[1]));
        gradient_matrix.loadFromArray(flat);
        delete[] flat;

        kernel_filter_gradient[i] = kernel_filter_gradient[i] + image.convolve(gradient_matrix);
    }
}

//Performs one itereation of gradient descent using the provided images, learning rate, and batch size
void ConvNet::descent(bool** images, char* labels, int num_images, float learning_rate, int epoch, int batch_size){
    if(!isConstructed) {
        throw std::invalid_argument("Unconstructed network");
        return;
    }

    int num_loops = num_images / batch_size;
    int sample_no;

    Matrix img(inputdim[0], inputdim[1]);
    
    //Hardcoded values
    Matrix correct(10, 1);
    correct = correct * 0;
    int cp = 0;
    for(int i = 0; i < num_loops; i++) {
        for(int j = 0; j < num_kernels; ++j) {
            kernel_filter_gradient[j] = kernel_filter_gradient[j] * 0;
            for(int k = 0; k < kernel_num_layers[j] - 1; ++k) {
                kernel_network_average_weights[j][k] = kernel_network_average_weights[j][k] * 0;
                kernel_network_average_biases[j][k] = kernel_network_average_biases[j][k] * 0;
            }
        }

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

            for(int k = 0; k < num_kernels; ++k) {
                for(int l = 0; l < kernel_num_layers[k] - 1; ++l) {
                    kernel_network_average_weights[k][l] = kernel_network_average_weights[k][l] + (kernel_network_activations[k][l] * (kernel_gradient[k][l + 1].transpose()));
                    kernel_network_average_biases[k][l] = kernel_network_average_biases[k][l] + kernel_gradient[k][l + 1];
                }
            }
        }
        
        float stepsize = learning_rate/batch_size;

        for(int j = 0; j < num_kernels; ++j) {
            kernel[j] = kernel[j] - (kernel_filter_gradient[j] * stepsize);
            for(int k = 0; k < kernel_num_layers[j] - 1; ++k) {
                kernel_network_weights[j][k] = kernel_network_weights[j][k] - (kernel_network_average_weights[j][k] * stepsize);
                kernel_network_biases[j][k] = kernel_network_biases[j][k] - (kernel_network_average_biases[j][k] * stepsize);
            }
        }
        for(int j = 0; j < num_layers - 1; j++) {
            weights[j] = weights[j] - (average_weights[j] * stepsize);
            biases[j] = biases[j] - (average_biases[j] * stepsize);
        }
        if((i * 100) / num_loops > cp) {
            cp += 1;
            std::cout << cp << "% | Epoch left: " << epoch << '\n';
        }
    }
}

//Saves the entire network to a file
//input_dims -> num_kernels -> kernel dimensions ->  kernel -> (kernel network) -> Numlayers -> layers sizes -> weights & biases
void ConvNet::writeToFile(const char* fname) {
    if(!isConstructed) {
        throw std::invalid_argument("Unconstructed network");
        return;
    }

    std::ofstream file(fname, std::ios::binary | std::ios::out);

    file.write((char*) &inputdim[0], sizeof(int));
    file.write((char*) &inputdim[1], sizeof(int));

    file.write((char*) &num_kernels, sizeof(int));
    float* temp = nullptr;

    for(int i = 0; i < num_kernels; ++i) {
        //Kernel dimensions
        file.write((char*) &kernel[i].dimension[0], sizeof(int));
        file.write((char*) &kernel[i].dimension[1], sizeof(int));

        //Kernel matrix
        temp = kernel[i].flatten();
        writeArray(file, temp, kernel[i].dimension[0] * kernel[i].dimension[1]);
        delete[] temp;
        temp = nullptr;

        //Kernel network
        //num layers and layer sizes
        file.write((char*) &kernel_num_layers[i], sizeof(int));
        writeArray(file, kernel_network_layers[i], kernel_num_layers[i]);

        //Kernel network weights and biases
        for(int j = 0; j < kernel_num_layers[i] - 1; ++j) {
            file.write((char*) &kernel_network_weights[i][j].dimension[0], sizeof(int));
            file.write((char*) &kernel_network_weights[i][j].dimension[1], sizeof(int));

            temp = kernel_network_weights[i][j].flatten();
            writeArray(file, temp, kernel_network_weights[i][j].dimension[0] * kernel_network_weights[i][j].dimension[1]);
            delete[] temp;

            file.write((char*) &kernel_network_biases[i][j].dimension[0], sizeof(int));
            file.write((char*) &kernel_network_biases[i][j].dimension[1], sizeof(int));

            temp = kernel_network_biases[i][j].flatten();
            writeArray(file, temp, kernel_network_biases[i][j].dimension[0] * kernel_network_biases[i][j].dimension[1]);
            delete[] temp;
            temp = nullptr;
        }
    }

    //Number of layers and layer sizes
    file.write((char*) &num_layers, sizeof(int));
    writeArray(file, layers, num_layers);

    //weights & biases, next to teach other
    for(int i = 0; i < (num_layers - 1); i++) {
        //Write dimensions of weight
        file.write((char*)&weights[i].dimension[0], sizeof(int));
        file.write((char*)&weights[i].dimension[1], sizeof(int));

        //Write the values
        temp = weights[i].flatten();
        writeArray(file, temp, weights[i].dimension[0] * weights[i].dimension[1]);
        delete[] temp;
        
        //------------------------------------

        //Write dimensions of bias
        file.write((char*)&biases[i].dimension[0], sizeof(int));
        file.write((char*)&biases[i].dimension[1], sizeof(int));

        //Write the values
        temp = biases[i].flatten();
        writeArray(file, temp, biases[i].dimension[0] * biases[i].dimension[1]);
        delete[] temp;
        temp = nullptr;
    }

    file.close();
}

//Loads the network from a file
//input_dims -> num_kernels -> kernel dimensions ->  kernel -> (kernel network) -> Numlayers -> layers sizes -> weights & biases
void ConvNet::loadFromFile(const char* fname) {
    std::ifstream file(fname, std::ios::binary | std::ios::in);
    
    float* tempf = nullptr;
    int* tempi = nullptr;

    file.read((char*) &inputdim[0], sizeof(int));
    file.read((char*) &inputdim[1], sizeof(int));

    file.read((char*) &num_kernels, sizeof(int));
    kernel = new Matrix[num_kernels];
    kernel_filter_gradient = new Matrix[num_kernels];
    kernel_network_weights = new Matrix*[num_kernels];
    kernel_network_biases = new Matrix*[num_kernels];
    kernel_network_layers = new int*[num_kernels];

    for(int i = 0; i < num_kernels; ++i) {
        //Destroy Kernel
        kernel[i].~Matrix();

        //Construct it
        tempi = readArrayInt(file, 2);
        kernel[i].createMatrix(tempi[0], tempi[1]);
        kernel_filter_gradient[i].createMatrix(tempi[0], tempi[1]);

        tempf = readArrayFloat(file, tempi[0] * tempi[1]);
        kernel[i].loadFromArray(tempf);

        delete[] tempf;
        delete[] tempi;
        tempf = nullptr;
        tempi = nullptr;

        file.read((char*) &kernel_num_layers[i], sizeof(int));
        delete[] kernel_network_layers[i];
        kernel_network_layers[i] = nullptr;
        kernel_network_layers[i] = readArrayInt(file, kernel_num_layers[i]);

        if(isConstructed) {
            delete[] kernel_network_weights[i];
            delete[] kernel_network_biases[i];
            kernel_network_weights[i] = nullptr;
            kernel_network_biases[i] = nullptr;
        }

        kernel_network_weights[i] = new Matrix[kernel_num_layers[i] - 1];
        kernel_network_biases[i] = new Matrix[kernel_num_layers[i] - 1];

        for(int j = 0; j < kernel_num_layers[i] - 1; ++j) {
            kernel_network_biases[i][j].~Matrix();
            kernel_network_weights[i][j].~Matrix();

            tempi = readArrayInt(file, 2);
            kernel_network_weights[i][j].createMatrix(tempi[0], tempi[1]);

            tempf = readArrayFloat(file, tempi[0] * tempi[1]);
            kernel_network_weights[i][j].loadFromArray(tempf);

            delete[] tempi;
            delete[] tempf;
            tempi = nullptr;
            tempf = nullptr;

            tempi = readArrayInt(file, 2);
            kernel_network_biases[i][j].createMatrix(tempi[0], tempi[1]);

            tempf = readArrayFloat(file, tempi[0] * tempi[1]);
            kernel_network_biases[i][j].loadFromArray(tempf);

            delete[] tempi;
            delete[] tempf;
            tempi = nullptr;
            tempf = nullptr;
        }
    }

    //load num of layers along with layer sizes
    file.read((char*) &num_layers, sizeof(int));
    delete[] layers;
    layers = nullptr;
    layers = readArrayInt(file, num_layers);
    
    if(isConstructed) {
        delete[] weights;
        delete[] biases;
        weights = nullptr;
        biases = nullptr;
    }
    
    weights = new Matrix[num_layers - 1];
    biases = new Matrix[num_layers - 1];

    //Read weights and biases in turn
    for(int i = 0; i < (num_layers - 1); i++) {
        //Destroy existing
        weights[i].~Matrix();
        biases[i].~Matrix();

        //Read dimensions and create matrix
        tempi = readArrayInt(file, 2);
        weights[i].createMatrix(tempi[0], tempi[1]);

        tempf = readArrayFloat(file, tempi[0] * tempi[1]);
        weights[i].loadFromArray(tempf);

        //Clearing memory
        delete[] tempf;
        delete[] tempi;
        tempf = nullptr;
        tempi = nullptr;

        //-------------------------------

        //Read dimensions and create matrix
        tempi = readArrayInt(file, 2);
        biases[i].createMatrix(tempi[0], tempi[1]);

        tempf = readArrayFloat(file, tempi[0] * tempi[1]);
        biases[i].loadFromArray(tempf);

        //Clearing memory
        delete[] tempf;
        delete[] tempi;
        tempf = nullptr;
        tempi = nullptr;
    }
    
    file.close();
}

//Loads the network configuration
void ConvNet::loadConfig(const char* fname) {
    std::ifstream file(fname, std::ios::in);
    //Find size of file
    file.seekg(0, std::ios::end);
    int len = file.tellg();

    //Load entire file to memore for parsing
    file.seekg(0);
    std::string f_str(len, '\0');
    file.read((char*)&f_str[0], sizeof(char) * len);
    
    //Extract properties from file
    std::string kernel_number = readProperty(f_str, "num_kernels");
    std::string layers_num = readProperty(f_str, "num_layers");
    std::string layers_val = readProperty(f_str, "layers");
    std::string kernel_layers_n = readProperty(f_str, "kernel_num_layers");
    std::string kernel_layers_v = readProperty(f_str, "kernel_layers");
    std::string kernel_dimensions_f = readProperty(f_str, "kernel_dimensions");
    std::string inputdims = readProperty(f_str, "input_dim");

    inputdim[0] = getInt(inputdims);
    inputdim[1] = getInt(inputdims);

    num_kernels = getInt(kernel_number);

    num_layers = getInt(layers_num) + 1;
    
    delete[] layers;

    layers = new int[num_layers];
    kernel_num_layers = new int[num_kernels];

    if(isConstructed) {
        delete[] weights;
        delete[] biases;
        weights = nullptr;
        biases = nullptr;
    }
    weights = new Matrix[num_layers - 1];
    biases = new Matrix[num_layers - 1];
    kernel = new Matrix[num_kernels];
    kernel_network_layers = new int*[num_kernels];
    kernel_network_weights = new Matrix*[num_kernels];
    kernel_network_biases = new Matrix*[num_kernels];

    layers[0] = 0;
    for(int i = 0; i < num_kernels; ++i) {
        kernel_num_layers[i] = getInt(kernel_layers_n) + 1;

        int rows = getInt(kernel_dimensions_f), cols = getInt(kernel_dimensions_f);
        kernel[i].createMatrix(rows, cols);

        kernel_network_layers[i] = new int[kernel_num_layers[i]];
        kernel_network_weights[i] = new Matrix[kernel_num_layers[i] - 1];
        kernel_network_biases[i] = new Matrix[kernel_num_layers[i] - 1];

        kernel_network_layers[i][0] = Matrix::getRowConvolve(inputdim[0], rows) * Matrix::getColConvolve(inputdim[1], cols);
        for(int j = 1; j < kernel_num_layers[i]; ++j) {
            kernel_network_layers[i][j] = getInt(kernel_layers_v);
            kernel_network_weights[i][j - 1].createMatrix(kernel_network_layers[i][j - 1],
            kernel_network_layers[i][j]);

            kernel_network_biases[i][j - 1].createMatrix(kernel_network_layers[i][j], 1);
        }

        layers[0] += kernel_network_layers[i][kernel_num_layers[i] - 1];
    }
    
    for(int i = 1; i < num_layers; i++) {
        layers[i] = getInt(layers_val);
        weights[i - 1].createMatrix(layers[i - 1], layers[i]);
        biases[i - 1].createMatrix(layers[i], 1);
    }

    file.close();
}
