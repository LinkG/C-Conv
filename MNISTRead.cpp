/*
    This class handles the reading of the binary image data and label data
    from the MNIST dataset of handwritten digits.
    Functions are described
*/
#include "MNISTRead.H"

/*
    Parameters: (int)Unreversed integer
    Because we read the binary data from left to right while the integers are
    defined with the leftmost being the highest power, we must reverse whatever
    binary data we read to get the actual integer
*/
int MNISTData::reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    
}

/*
    Parameters:(ifstream) MNIST label file
    returns the first integer that identifies MNIST datasets
*/
int MNISTData::getMagicNum(std::ifstream& file) {
    int magic = 0;
    file.read((char*) &magic, sizeof(int));
    magic = reverseInt(magic);
    return magic;
}

/*
    Constructor:(string) Path to image file, (string) Path to label file
    Sets up the input streams
*/
MNISTData::MNISTData(const std::string images, const std::string labels, int num) {
    MNISTData::labels.open(labels, std::ios::binary);
    MNISTData::images.open(images, std::ios::binary);
    number_imgs = num;
}

//Destructor
MNISTData::~MNISTData() {
    images.close();
    labels.close();
}

/*
    Parameters: (int) Image size
    Reads from the images data, and stores the number of images and size
    of image int the parameter variables. It returns a 2D array of unsigned chars
    that have the images. An image as an array of unsigned char that is either binary '0'
    or binary '1' depending in if the pixel is white or dark.
*/
int MNISTData::getImages(int row_in, int col_in, bool** store) {
    if(images.is_open()) {
        int num_images = 0;
        if(getMagicNum(images) != 2051) std::runtime_error("INVALID IMAGES FILE");
        int rows, col;
        images.read((char*)&num_images, sizeof(int)), num_images = reverseInt(num_images);
        images.read((char*)&rows, sizeof(int)), rows = reverseInt(rows);
        images.read((char*)&col, sizeof(int)), col = reverseInt(col);
        int img_size = rows * col;
        if(number_imgs == -1 || number_imgs > num_images || row_in != rows || col_in != col) {
            std::runtime_error("Mismathcing array size");
            return -1;
        }
        uchar** _data = new uchar*[number_imgs];
        for(l = 0; l < number_imgs; l++) {
            _data[l] = new uchar[img_size];
            images.read((char*)_data[l], img_size);
        }
        for(l = 0; l < number_imgs; l++) {
            for(int k = 0; k < img_size; k++) {
                store[l][k] = _data[l][k] ? true : false;
            }
            delete[] _data[l];
        }
        delete[] _data;
        return 1;
    } else {
        std::runtime_error("Cant open file images");
        return 0;
    }
}

/*
    Parameters:(int) Number of labels
    The function reads the label data and returns a unsigned char array of binary integers,
    i.e. they have to be cast to int before use
*/
int MNISTData::getLabels(char* store) {
    if(labels.is_open()) {
        int num_labels = 0;
        if(getMagicNum(labels) != 2049) std::runtime_error("INVALID LABELS FILE");
        labels.read((char*) &num_labels, sizeof(num_labels)), num_labels = reverseInt(num_labels);
        if(number_imgs == -1 || number_imgs > num_labels) {
            std::runtime_error("Mismatching array size");
            return -1;
        }
        char tmp;
        for(l = 0; l < number_imgs; l++) {
            labels.read((char*)&tmp, 1);
            store[l] = char(tmp + '0');
        }
    } else {
        std::runtime_error("Cant open file labels");
        return -1;
    }
    return 0;
}