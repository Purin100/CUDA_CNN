#pragma once
#include "utils.h"
/*
Pooling type from Pooling.h
 MAXPOOL 0
 MINPOOL 1
 NORMALPOOL 2
*/

enum LayerType
{
    UNKNOWN = 0,
    DENSE = 1,
    CONV2D = 2,
    MAXPOOLING = 3,
    AVERAGEPOOLING = 4,
    FLATTEN = 5,
    DROPOUT = 6,
    //LASTGUARD
};

struct LayerInfo
{
    int input_units = 0, output_units = 0;
    int input_rows = 0, input_cols = 0;
    int offset = -1;

    int kernel_rows, kernel_cols;
    int stride_row, stride_col;
    int channels = -1;
    int layer_units = 0;//how many neurons in this layer. It applies to Neuron(Dense), Conv2D, Maxpooling and Averagepooling objects.

    float coff = 0.0f;

    bool isDistinguish = false;
    bool isDropout = false;
    const char* activation = nullptr;
    const char* padding = "valid";
    char poolmode = 2;
};

struct DenseInfo
{
    DenseInfo(int input_units = 0, int output_units = 0,
        int layer_units = 0,
        MYTYPE drop_rate = 0.0f,
        bool use_bias = false,
        const char* activation = nullptr,
        MYTYPE _lr = 0.0f)
    {
        this->input_units = input_units, this->output_units = output_units;
        this->layer_units = layer_units;
        this->drop_rate = drop_rate;
        this->use_bias = use_bias;
        this->activation = activation;
        lr = _lr;
    }

    int input_units = 0, output_units = 0;
    int layer_units = 0;
    MYTYPE drop_rate = 0.0f;
    MYTYPE lr = 0.0f;
    bool use_bias = false;
    const char* activation = nullptr;
};

struct Conv2DInfo
{
    /*bool Conv2D::BuildLayer(int input_rows, int input_cols,
        int kernel_rows, int kernel_cols,
        int stride_row, int stride_col,
        int layer_units,
        int channels,
        const char* activation,
        const char* padding)*/
    Conv2DInfo(int input_rows = 0, int input_cols = 0,
        int kernel_rows = 0, int kernel_cols = 0,
        int stride_row = -1, int stride_col = -1,
        int layer_units = 0,//how many neurons in this layer.
        int channels = -1,
        const char* activation = nullptr,
        const char* padding = "valid",
        bool isDropout = false)
    {
        this->input_rows = input_rows, this->input_cols = input_cols;
        this->kernel_rows = kernel_rows, this->kernel_cols = kernel_cols;
        this->stride_row = stride_row, this->stride_col = stride_col;
        this->channels = channels;
        this->layer_units = layer_units;
        this->isDropout = isDropout;
        this->activation = activation;
        this->padding = padding;
    }

    int input_rows = 0, input_cols = 0;
    int kernel_rows = 0, kernel_cols = 0;
    int stride_row, stride_col;
    int channels = -1;
    int layer_units = 0;//how many neurons in this layer. It applies to Neuron(Dense), Conv2D, Maxpooling and Averagepooling objects.
    bool isDropout = false;
    const char* activation = nullptr;
    const char* padding = "valid";
};

/*bool BuildLayer(int input_units,
                    int input_rows, int input_cols,
                    int kernel_rows, int kernel_cols,
                    int stride,
                    char pool_mode = NORMALPOOL);*/
struct MaxpoolInfo
{
    MaxpoolInfo(int input_units = 0,
        int input_rows = 0, int input_cols = 0,
        int kernel_rows = 0, int kernel_cols = 0,
        int stride_row = -1, int stride_col = -1,
        poolMode mode = NORMALPOOL)
    {
        this->input_units = input_units;
        this->input_rows = input_rows, this->input_cols = input_cols;
        this->kernel_rows = kernel_rows, this->kernel_cols = kernel_cols;
        this->stride_row = stride_row;
        this->stride_col = stride_col;
        this->poolmode = mode;
    }

    int input_units = 0;
    int input_rows = 0, input_cols = 0;
    int kernel_rows = 0, kernel_cols = 0;
    int stride_row = -1, stride_col = -1;
    poolMode poolmode = NORMALPOOL;
};

struct Net_LayerInfo
{
    Net_LayerInfo() {};
    ~Net_LayerInfo()
    {
        layer = nullptr;
    }
    Net_LayerInfo(void* p, int num, LayerType type, int order)
    {
        this->layer = p;
        this->num = num;
        this->type = type; //>= LayerType::DENSE && type < LayerType::LASTGUARD ? type : 0;
        this->order = order;
    }
    Net_LayerInfo& operator=(const Net_LayerInfo& _n)
    {
        if (this != &_n)
        {
            this->layer = _n.layer;
            this->num = _n.num;
            this->type = _n.type;
            this->order = _n.order;
        }
        return *this;
    }

    void* layer = nullptr;
    int num = 0;
    LayerType type = UNKNOWN;
    int order = 0;
};

struct FlattenInfo
{
    FlattenInfo(int in_row, int in_col)
    {
        input_row = in_row;
        input_col = in_col;
    }
    int input_row, input_col;
};
