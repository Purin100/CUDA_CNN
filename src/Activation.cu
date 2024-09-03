#include "Activation.cuh"

__device__ MYTYPE devTanh(MYTYPE x)
{
    return tanh(x);
}
__device__ MYTYPE devTanhGra(MYTYPE x)
{
    return (1.0 - tanh(x) * tanh(x));
}

__device__ MYTYPE devRelu(MYTYPE x)
{
    return x > 0.0 ? x : 0.0;
}
__device__ MYTYPE devReluGra(MYTYPE x)
{
    return x > 0.0;
}

__device__ MYTYPE devSigmoid(MYTYPE x)
{
    return (1.0 / (1.0 + exp(-x)));
}
__device__ MYTYPE devSigmoid_grad(MYTYPE x)
{
    return exp(-x) / ((1.0 + exp(-x)) * (1.0 + exp(-x)));
}

__global__ void Tanh(MYTYPE* x, MYTYPE* output, const int num)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num)
        output[i] = tanh(x[i]);
}
__global__ void Tanh_gradient(MYTYPE* x, MYTYPE* output, const int num)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num)
        output[i] = devTanhGra(x[i]);
}


__global__ void Relu1(MYTYPE* x, MYTYPE* output, const int num)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num)
    {
        if (x[i] > 1.0)
            output[i] = 1.0;
        else
            output[i] = x[i] > 0.0 ? x[i] : 0.0;
    }
}
__global__ void Relu1_gradient(MYTYPE* x, MYTYPE* output, const int num)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num)
        output[i] = (x[i] > 0.0);
}


__global__ void Relu(MYTYPE* x, MYTYPE* output, const int num)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num)
        output[i] = x[i] > 0.0 ? x[i] : 0.0;
}
__global__ void Relu_gradient(MYTYPE* x, MYTYPE* output, const int num)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num)
        output[i] = (x[i] > 0.0);
}

__global__ void LeakyRelu(MYTYPE* x, MYTYPE* output, const int num)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num)
        output[i] = x[i] > 0.0 ? x[i] : 0.01 * x[i];
}
__global__ void LeakyRelu_gradient(MYTYPE* x, MYTYPE* output, const int num)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num)
        output[i] = x[i] > 0.0 ? 1.0 : 0.01;
}

__global__ void Sigmoid(MYTYPE* x, MYTYPE* output, const int num)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num)
        output[i] = devSigmoid(x[i]);
}
__global__ void Sigmoid_gradient(MYTYPE* x, MYTYPE* output, const int num)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num)
        output[i] = devSigmoid_grad(x[i]);
}

__global__ void Linear(MYTYPE* x, MYTYPE* output, const int num)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num)
        output[i] = x[i];
}
__global__ void Linear_gradient(MYTYPE* x, MYTYPE* output, const int num)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num)
        output[i] = 1.0;
}

//__global__ void Softmax(MYTYPE* x, MYTYPE* output, const int num)
//{
//    int i = threadIdx.x + blockDim.x * blockIdx.x;
//    int tid = threadIdx.x;
//    __shared__ MYTYPE sum[10];
//
//    if (i < num)
//        sum[i] = exp(x[i]);
//    __syncthreads();
//
//    if (i < num)
//    {
//        for (int index = 1; index < blockDim.x; index *= 2)
//        {
//            if (threadIdx.x % (index * 2) == 0)
//            {
//                sum[tid] += (sum[tid + index]);
//            }
//            __syncthreads();
//        }
//    }
//    __syncthreads();
//   /* if (threadIdx.x == 0)
//    {
//        output[0] = sum[0];
//    }*/
//    output[i] = exp(x[i]) / sum[0];
//
//}

MYTYPE Cross_Entropy(MYTYPE* one_hot, MYTYPE* x, const int classnum, const int sample_num)
{
    MYTYPE loss = 0.0f;
    //float sum = 0.f;
    for (int i = 0; i < classnum; i++)
    {
        loss += (one_hot[i] * log(x[i]));
    }
    loss = -loss;
    return sample_num == 1 ? loss : loss / sample_num;
}

MYTYPE MSE(MYTYPE* one_hot, MYTYPE* x, const int classnum)
{
    MYTYPE loss = 0.0;
    for (int i = 0; i < classnum; i++)
        loss += (x[i] - one_hot[i]) * (x[i] - one_hot[i]);
    return loss * 0.5;
}