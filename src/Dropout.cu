#include "Dropout.h"
#include <ctime>

__global__ void RandVec(MYTYPE* vec, const int size, const MYTYPE drop_rate, curandState* state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
    {
        vec[i] = curand_uniform_double(&state[i]);
        if (vec[i] < drop_rate)
            vec[i] = 0.0;
        else
            vec[i] = 1.0;
    }
}

__global__ void Setup(curandState* state, unsigned long seed, int size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
        curand_init(seed, i, 0, &state[i]);
}

bool Dropout::BuildLayer(const int input_units, MYTYPE drop_rate)
{
    if (input_units <= 0)
    {
        printf("ERROR: invalid parament input_units in Dropout::BuildLayer(const int, MYTYPE). input_units should larger than zero.\n");
        getchar();
        return false;
    }

    this->input_unit = input_units;
    this->output_unit = input_units;
    this->drop_rate = drop_rate > 0.0 && drop_rate < 1.0 ? drop_rate : 0.5;
    scale = (1.0 - this->drop_rate);
    output = Vector(output_unit);
    randnum = Vector(output_unit);
    loss = Vector(output_unit);
    cudaMalloc((void**)&state, sizeof(curandState) * input_units);
    int threads = 512, blocks = (input_units + threads - 1) / threads;
    Setup << <blocks, threads >> > (state, time(NULL), input_units);
    cudaDeviceSynchronize();
    return true;
}

void Dropout::Forward(Vector& pre_layer, bool train)
{
    if (pre_layer.empty())
    {
        printf("ERROR: invalid parament pre_layer (empty vector) in Dropout::Forward(Dense*).\n");
        getchar();
        return;
    }

    int threads = 512;
    int blocks = (input_unit + threads - 1) / threads;
    output = pre_layer;
    //if (train)
    {
        RandVec << <blocks, threads >> > (randnum.GetDevVec(), input_unit, drop_rate, state);
        cudaDeviceSynchronize();
        mc.VecEleMult(output.GetDevVec(), randnum.GetDevVec(), output_unit);
        mc.VecMultNum(output.GetDevVec(), scale, output_unit);
    }
}

void Dropout::Backward(Vector& loss)
{
    if (loss.empty())
    {
        printf("ERROR: invalid parament pre_layer (empty vector) in Dropout::Forward(Dense*).\n");
        getchar();
        return;
    }
    //output = loss;
    mc.VecEleMult(this->loss.GetDevVec(), randnum.GetDevVec(), output_unit);
    this->loss.DataTransfer(DeviceToHost);
    mc.VecMultNum(this->loss.GetDevVec(), 1.0 / scale, output_unit);
    loss = this->loss;
}