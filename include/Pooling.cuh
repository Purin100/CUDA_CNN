#ifndef __POOLING_H__
#define __POOLING_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <math.h>
#include "utils.h"
#include "Matrix.cuh"



struct Position
{
    Position() { x = -1, y = -1; };
    Position(int _x, int _y)
    {
        this->x = _x;
        this->y = _y;
    }
    Position& operator=(const Position& _pos)
    {
        if (&_pos != this)
        {
            this->x = _pos.x;
            this->y = _pos.y;
        }
        return *this;
    }

    int x;
    int y;
};

class Maxpooling
{
public:
    Maxpooling();
    ~Maxpooling();
    //if useMinpool sets true, the pooling layer will filter max(positive) and min(negative) values for inputs seperately
    //otherwise pooling layer will only filter values larger than zero.
    bool BuildLayer(int input_units,
        int input_rows, int input_cols, int input_channels,
        int kernel_rows, int kernel_cols,
        int stride_x, int stride_y,
        poolMode pool_mode = NORMALPOOL);

    void Forward(Matrix& input);
    //Matrix pre_layer should be the loss matrix in a Conv2D layer
    void Backward(Matrix& pre_layer_loss);

    void Save(std::string& _dir, int which);

    Matrix& Getoutput(){return output;}
    void GetIndex(int* idx);
    int GetOutputRow() { return output.rows(); }
    int GetOutputCol() { return output.cols(); }
    int OutRow(){return out_rows;}
    int OutCol(){return out_cols;}
    int size(){return out_cols * out_rows * units;}
    int unit(){return units;}
    int k_row() { return kernel_rows; }
    int k_col() { return kernel_cols; }
    int s_row() { return stride_x; }
    int s_col() { return stride_y; }
    poolMode GetType() { return type; }
    // void Forward(Conv2D* _input);
    Matrix loss;
private:
    int kernel_rows, kernel_cols;
    int stride_x, stride_y, units;
    int out_rows, out_cols;
    int input_rows, input_cols;
    //std::vector<Position> max_pos;
    Matrix output;
    int* index = nullptr;
    poolMode type;
};
typedef Maxpooling* pMaxp;

#endif