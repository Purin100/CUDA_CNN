#pragma once
#include <stdio.h>
#include <algorithm>
#include <random>
#include "utils.h"
#include "Activation.cuh"
#include "TXTReader.h"
#include "Matrix.cuh"
#include "im2col.h"
#include "col2im.h"
#include "Pooling.cuh"
#include "optimizer.h"

enum paddingModes
{
    VALID=0,SAME=1
};

class Conv2D
{
public:
    Conv2D();
    ~Conv2D() 
    {
        if (adam)
        {
            delete adam;
            adam = nullptr;
        }
    };

    bool BuilderLayer(int input_row, int input_col, int k_row, int k_col,
        int stride_row, int stride_col, int channels, int units, int input_units,
        const char* padding_mode, const char* activation, const char* optimizer = nullptr);
    void Forward(Conv2D* pre_layer);
    void Forward(pMaxp pre_layer);
    void Forward(MYTYPE* _input, int row, int col);
    void Forward(MYTYPE** _input, int row, int col);
    void Forward(Vector& _input, int row, int col, int channels);

    void Backward(Conv2D* pre_layer, bool update, const int batch_size);
    void Backward(Maxpooling* pre_layer, bool update, const int batch_size);
    void Backward(bool update, const int batch_size);

    void lrDecay(const int now_epoch);

    int GetOutputRow() { return output.rows(); }
    int GetOutputCol() { return output.cols(); }
    int OutR() { return output_row; }
    int OutC() { return output_col; }
    int GetUnits() { return units; }
    int GetChannels() { return channels; }

    void showOutput() { output.showMat(); }
    void Save(string& _dir, int which);
    void Setlr(MYTYPE _lr) { this->lr = (_lr > 0.0) ? _lr : 1e-3; }

    Matrix& GetOutput() { return output; }

    Matrix loss;
    Matrix loss_colImg;
private:
    int kernel_row = 0, kernel_col = 0;
    int input_row, input_col, output_row, output_col;
    int padding_rows, padding_cols;
    int stride_rows, stride_cols;
    int channels;
    int units, input_units = 0;
    int kernel_units = 0;//for im2col
    int adj_row = 0, adj_col = 0;//for col2img, ensure the size of reformated matrix is the same as loss matrix in previous layer
    int sample_num = 0;
    MYTYPE lr = 1e-3;
    MYTYPE base_threshold = 0.0;

    Matrix kernel, kernel_grad, kernel_t;
    Matrix grad_sample, save_grad;
    Matrix input, colImg, output, local_out;
    Matrix differial;
    Matrix colImg_t;    

    //activation function and its gradient
    void (*activation)(MYTYPE* input, MYTYPE* output, const int size) = nullptr;
    void (*gradient)(MYTYPE* input, MYTYPE* output, const int size) = nullptr;
    Adam* adam = nullptr;
    char padding_mode;
protected:
    //im2col, this function doesn't support implace (src cannot be the same matrix as dst).
    //src_row is the output rows (pre_layer->output_row) in previous layer.src_col is the output columns (pre_layer->output_col) in previous layer.
    //e.g., you get 32 (pre_layer->units) * 28 (pre_layer->output_row) * 28 (pre_layer->output_col), the shape of pre_layer->output is 32 * 784
    //DO NOT set src_row=32 and src_col=784. YOU SHOULD SET src_row=28 and src_col=28.
    void _im2col(Matrix& src, Matrix& dst, int src_row, int src_col, int kernel_row, int kernel_col, int kernel_units/*channels*/,
        int stride_row, int stride_col, int padding_row, int padding_col);

    void _col2im(Matrix& src, Matrix& dst, int src_row, int src_col, int kernel_row, int kernel_col, int kernel_units/*channels*/,
        int stride_row, int stride_col, int padding_row, int padding_col);
};