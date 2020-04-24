#ifndef MNISTRead
#define MNISTRead
#include <fstream>
typedef unsigned char uchar;
class MNISTData {
    int l;
    int reverseInt(int);
    int number_imgs;
    std::ifstream labels;
    std::ifstream images;
    int getMagicNum(std::ifstream&);
    public:
    MNISTData(const char *, const char *, int);
    char* getLabels();
    float** getImages(int&);
    ~MNISTData();
};

#endif