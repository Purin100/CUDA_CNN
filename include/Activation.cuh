#pragma once

#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "utils.h"

__global__ void Tanh(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Relu(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Relu1(MYTYPE* x, MYTYPE* output, const int num);//max return value is one
__global__ void Sigmoid(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Linear(MYTYPE* x, MYTYPE* output, const int num);
__global__ void LeakyRelu(MYTYPE* x, MYTYPE* output, const int num);

__global__ void Tanh_gradient(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Relu_gradient(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Relu1_gradient(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Linear_gradient(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Sigmoid_gradient(MYTYPE* x, MYTYPE* output, const int num);
__global__ void LeakyRelu_gradient(MYTYPE* x, MYTYPE* output, const int num);

//__global__ void Softmax(MYTYPE* x, MYTYPE* output, const int num);
MYTYPE Cross_Entropy(MYTYPE* one_hot, MYTYPE* x, const int classnum, const int sample_num = 1);
MYTYPE MSE(MYTYPE* one_hot, MYTYPE* x, const int classnum);

static void Softmax(MYTYPE* x, MYTYPE* result, int classNum)
{
    if (!x)
        return;
    if (!result)
        return;

    MYTYPE sum = 0.0, max=x[0];

    //for (int i = 1; i < classNum; i++)
    //    if (x[i] > max)
    //        max = x[i];

    for (int i = 0; i < classNum; i++)
    {
        if (isnan(x[i]))
        {
            printf("nan dected in softmax (input vector). %d\n", i);
            getchar();
        }
        //x[i] -= max;
        sum += exp(x[i]);
    }

    for (int i = 0; i < classNum; i++)
        result[i] = exp(x[i]) / sum;

}
