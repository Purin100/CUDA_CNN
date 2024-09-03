#pragma once
#include "utils.h"
#include "Matrix.cuh"
#include <opencv.hpp>
#include <string>

class PNGReader
{
public:
    PNGReader();
    PNGReader(_In_ cv::Mat& pic);
    ~PNGReader();

    /*
    Arg:
    read_mode: should be "grey" or "colour"
    */
    bool ReadFile(_In_ const char* read_mode, _In_ const char* _filename);
    bool OpenFileInDir(_In_ std::string _dir);

    void Shrink(ShrinkMode _mode);

    //char success;
    Vector pixel;
    //Matrix img;
    int label;
private:
    int width, height, channels;
};