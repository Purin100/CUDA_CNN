#pragma once
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>
#include "utils.h"
#include "Matrix.cuh"
#include "Mat_calc.cuh"
#pragma comment(lib,"curand.lib")

class Dropout
{
public:
    Dropout() {};
    ~Dropout() { cudaFree(state); };

    bool BuildLayer(const int input_units, MYTYPE drop_rate);

    void Forward(Vector& pre_layer, bool train=false);
    void Backward(Vector& loss);

    Vector& GetOutput() { return output; }
    int Getsize() { return output_unit; }

    Vector loss;
private:
    curandState* state = nullptr;
    Vector output;
    Vector randnum;
    int input_unit = 0, output_unit = 0;
    MYTYPE drop_rate = 0.5;
    MYTYPE scale = 0.0;
};