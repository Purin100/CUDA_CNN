#pragma once
#include <cuda_runtime.h>
//#include "curand.h"
#include <cublas_v2.h>
#include "utils.h"
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math.h>


__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, MYTYPE* input, MYTYPE* output, int* indexes);

__global__ void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, MYTYPE* delta, MYTYPE* prev_delta, int* indexes);