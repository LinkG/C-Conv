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
    file.read((char*) &ret, sizeof(int));
    return ret;
}

float* readArrayFloat(std::ifstream &file, int size) {
    float* ret = new float[size];
    file.read((char*) &ret, sizeof(float));
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
    layer_0_dimensions[0] = Matrix::getRowConvolve(img_size, kernel.dimension[0]);
    layer_0_dimensions[1] = Matrix::getColConvolve(img_size, kernel.dimension[1]);
    layers[0] =  layer_0_dimensions[0] * layer_0_dimensions[1];

    weights = new Matrix[num_layers - 1];
    biases = new Matrix[num_layers - 1];
    average_weights = new Matrix[num_layers - 1];
    average_biases = new Matrix[num_layers - 1];
    activations = new Matrix[num_layers];
    gradients = new Matrix[num_layers];
    pre_sigmoid = new Matrix[num_layers];

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

    isConstructed = true;
}

//Gives random values to all weights, biases and kernel using a normal distribuition
void ConvNet::makeCNNRandom() {
    if(!isConstructed) {
        throw std::invalid_argument("Unconstructed network");
        return;
    }

    kernel.doFunction(generateRandom);
    
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

    Matrix convolved(layer_0_dimensions[0], layer_0_dimensions[1]);

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
    if(!isConstructed) {
        throw std::invalid_argument("Unconstructed network");
        return;
    }

    gradients[num_layers - 1] = (activations[num_layers - 1] - correct) ^ pre_sigmoid[num_layers - 1];
    for(int i = num_layers - 2; i >= 0; i--) {
        gradients[i] = (weights[i] * gradients[i + 1]) ^ pre_sigmoid[i];
    }

    float* flat  = gradients[0].flatten();
    
    Matrix gradient_matrix(layer_0_dimensions[0], layer_0_dimensions[1]);
    gradient_matrix.loadFromArray(flat);

    delete[] flat;

    kernel_gradient = kernel_gradient + image.convolve(gradient_matrix);
}

//Performs one itereation of gradient descent using the provided images, learning rate, and batch size
void ConvNet::descent(float** images, char* labels, int num_images, float learning_rate, int batch_size){
    if(!isConstructed) {
        throw std::invalid_argument("Unconstructed network");
        return;
    }

    int num_loops = num_images / batch_size;
    int sample_no;

    Matrix img(img_size, img_size);
    
    //Hardcoded values
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

//Saves the entire network to a file
//kernel dimensions ->  kernel -> Numlayers -> layers sizes -> weights & biases
void ConvNet::writeToFile(const char* fname) {
    if(!isConstructed) {
        throw std::invalid_argument("Unconstructed network");
        return;
    }

    std::ofstream file(fname, std::ios::binary | std::ios::out);

    float* temp = nullptr;

    //Kernel dimensions
    file.write((char*) &kernel.dimension[0], sizeof(int));
    file.write((char*) &kernel.dimension[1], sizeof(int));

    //kernel matrix
    temp = kernel.flatten();
    writeArray(file, temp, kernel.dimension[0] * kernel.dimension[1]);
    delete[] temp;
    temp = nullptr;

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
//kernel dimensions ->  kernel -> Numlayers -> layers sizes -> weights & biases
void ConvNet::loadFromFile(const char* fname) {
    std::ifstream file(fname, std::ios::binary | std::ios::in);
    
    float* tempf = nullptr;
    int* tempi = nullptr;
    //Destroying existing kernel
    kernel.~Matrix();

    //Loading dimensions
    tempi = readArrayInt(file, 2);
    kernel.createMatrix(tempi[0], tempi[1]);

    //Loading kernel
    tempf = readArrayFloat(file, tempi[0] * tempi[1]);
    kernel.loadFromArray(tempf);

    //Clearing memory
    delete[] tempf;
    delete[] tempi;
    tempf = nullptr;
    tempi = nullptr;

    //load num of layers along with layer sizes
    file.read((char*) &num_layers, sizeof(int));
    delete[] layers;
    layers = nullptr;
    layers = readArrayInt(file, num_layers);

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
    file.seekg(0, std::ios::end);
    int len = file.tellg();
    file.seekg(0);
    std::string f_str(len, '\0');
    file.read((char*)&f_str[0], sizeof(char) * len);
    
    std::string kernel_dimensions = readProperty(f_str, "kernel");
    std::string layers_num = readProperty(f_str, "num_layers");
    std::string layers_val = readProperty(f_str, "layers");

    kernel.createMatrix(getInt(kernel_dimensions), getInt(kernel_dimensions));
    kernel_gradient.createMatrix(getInt(kernel_dimensions), getInt(kernel_dimensions));

    num_layers = getInt(layers_num) + 1;
    
    delete[] layers;

    layers = new int[num_layers];

    for(int i = 1; i < num_layers; i++) {
        layers[i] = getInt(layers_val);
    }

    file.close();
}
