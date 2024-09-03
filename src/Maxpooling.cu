/*
Most of functions in this file are from https://github.com/pjreddie/darknet/blob/master/src/maxpool_layer_kernels.cu
Some changes are made for this project
*/
#include "Pooling.cuh"

Maxpooling::Maxpooling()
{

}

Maxpooling::~Maxpooling()
{
    if (index)
        cudaFree(index);
}

bool Maxpooling::BuildLayer(int input_units,
    int input_rows, int input_cols, int input_channels,
    int kernel_rows, int kernel_cols,
    int stride_x, int stride_y,
    poolMode pool_mode)
{
    if (input_units <= 0)
    {
        printf("(in Maxpooling BuildLyaer) ERROR: argument input_units should be larger than zero.\n");
        getchar();
        return false;
    }


    if (input_rows <= 0 || input_cols <= 0)
    {
        printf("(in Maxpooling BuildLyaer) ERROR: argument input_rows and input_cols should be larger than zero.\n");
        getchar();
        return false;
    }
    if (kernel_rows <= 0 || kernel_cols <= 0)
    {
        printf("(in Maxpooling BuildLyaer) ERROR: argument kernel_rows and kernel_cols should be larger than zero.\n");
        getchar();
        return false;
    }
    this->kernel_rows = kernel_rows;
    this->kernel_cols = kernel_cols;
    this->stride_x = stride_x > 0 ? stride_x : 1;
    this->stride_y = stride_y > 0 ? stride_y : 1;

    //if (input_units == 1)
    //{
    //    this->input_rows = input_channels;
    //    this->input_cols = input_rows * input_cols;
    //}
    //else
    {
        this->input_rows = input_rows;
        this->input_cols = input_cols;
    }

    this->units = input_units;

    switch (pool_mode)
    {
    case NORMALPOOL:
        type = NORMALPOOL;
        break;
    default:
        type = NORMALPOOL;
        break;
    }

    if (input_rows != 0 && input_cols != 0)
    {
        out_rows = (this->input_rows - this->kernel_rows) / this->stride_x + 1;
        out_cols = (this->input_cols - this->kernel_cols) / this->stride_y + 1;
        if (input_channels == 1)
            output = Matrix(out_rows, out_cols);
        else
            output = Matrix(input_channels, out_rows * out_cols);
        cudaMalloc((void**)&index, sizeof(int) * output.size());
    }
    if (input_channels == 1)
        loss = Matrix(out_rows, out_cols);
    else
        loss = Matrix(input_channels, out_rows * out_cols);
    return true;
}




__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_h, int stride_w, int kernel_h, int kernel_w,
    int pad, MYTYPE* input, MYTYPE* output, int* indexes)
{
    int h = (in_h + pad - kernel_h) / stride_h + 1;
    int w = (in_w + pad - kernel_w) / stride_w + 1;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad / 2;
    int h_offset = -pad / 2;

    int out_index = j + w * (i + h * (k + c * b));
    MYTYPE max = -HUGE_VAL;
    int max_i = -1;
    int l, m;
    for (l = 0; l < kernel_h; ++l) {
        for (m = 0; m < kernel_w; ++m) {
            int cur_h = h_offset + i * stride_h + l;
            int cur_w = w_offset + j * stride_w + m;
            int index = cur_w + in_w * (cur_h + in_h * (k + b * in_c));
            int valid = (cur_h >= 0 && cur_h < in_h&&
                cur_w >= 0 && cur_w < in_w);
            MYTYPE val = (valid != 0) ? input[index] : -HUGE_VAL;
            max_i = (val > max) ? index : max_i;
            max = (val > max) ? val : max;
        }
    }
    output[out_index] = max;
    indexes[out_index] = max_i;
}

//__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, MYTYPE* input, MYTYPE* output, int* indexes)
__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_h, int stride_w, int size, int pad, MYTYPE* input, MYTYPE* output, int* indexes)
{
    /*int h = (in_h + pad - size) / stride + 1;
    int w = (in_w + pad - size) / stride + 1;*/
    int h = (in_h + pad - size) / stride_h + 1;
    int w = (in_w + pad - size) / stride_w + 1;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad / 2;
    int h_offset = -pad / 2;

    int out_index = j + w * (i + h * (k + c * b));
    MYTYPE max = -HUGE_VAL;
    int max_i = -1;
    int l, m;
    for (l = 0; l < size; ++l) {
        for (m = 0; m < size; ++m) {
            int cur_h = h_offset + i * stride_h + l;
            int cur_w = w_offset + j * stride_w + m;
            int index = cur_w + in_w * (cur_h + in_h * (k + b * in_c));
            int valid = (cur_h >= 0 && cur_h < in_h&&
                cur_w >= 0 && cur_w < in_w);
            MYTYPE val = (valid != 0) ? input[index] : -HUGE_VAL;
            max_i = (val > max) ? index : max_i;
            max = (val > max) ? val : max;
        }
    }
    output[out_index] = max;
    indexes[out_index] = max_i;
}
void Maxpooling::Forward(Matrix& input)
{
    int n = output.size();
    UINT k = (n - 1) / BLOCK + 1;
    UINT x = k;
    UINT y = 1;
    if (x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK) + 1;
    }
    dim3 blocks = { x, y, 1 };
    forward_maxpool_layer_kernel <<<blocks, BLOCK>>> (n, input_rows, input_cols, units,
        stride_x, stride_y, kernel_rows, kernel_cols, 0,
        input.GetDevMat(), output.GetDevMat(), index);
    cudaDeviceSynchronize();
    output.DataTransfer(DeviceToHost);
}



__global__ void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_h, int stride_w, int kernel_h, int kernel_w,
    int pad, MYTYPE* delta, MYTYPE* prev_delta, int* indexes)
{
    int h = (in_h + pad - kernel_h) / stride_h + 1;
    int w = (in_w + pad - kernel_w) / stride_w + 1;
    int c = in_c;
    int area_h = (kernel_h - 1) / stride_h;
    int area_w = (kernel_w - 1) / stride_w;

    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;

    int index = id;
    int j = id % in_w;
    id /= in_w;
    int i = id % in_h;
    id /= in_h;
    int k = id % in_c;
    id /= in_c;
    int b = id;

    int w_offset = -pad / 2;
    int h_offset = -pad / 2;

    MYTYPE d = 0;
    for (int l = -area_h; l <= area_h; l++)
        for (int m = -area_w; m <= area_w; m++)
        {
            int out_w = (j - w_offset) / stride_w + m;
            int out_h = (i - h_offset) / stride_h + l;
            int out_index = out_w + w * (out_h + h * (k + c * b));
            int valid = (out_w >= 0 && out_w < w&&
                out_h >= 0 && out_h < h);
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
        }
    prev_delta[index] += d;
}

// If the stride of x and y are different, use this function
// stride_w means stride on y axis
// stride_h means stride on x axis
__global__ void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_h, int stride_w, int size, int pad, MYTYPE* delta, MYTYPE* prev_delta, int* indexes)
{
    int h = (in_h + pad - size) / stride_h + 1;
    int w = (in_w + pad - size) / stride_w + 1;
    int c = in_c;
    int area_h = (size - 1) / stride_h;
    int area_w = (size - 1) / stride_w;

    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;

    int index = id;
    int j = id % in_w;
    id /= in_w;
    int i = id % in_h;
    id /= in_h;
    int k = id % in_c;
    id /= in_c;
    int b = id;

    int w_offset = -pad / 2;
    int h_offset = -pad / 2;

    MYTYPE d = 0;
    for(int l = -area_h; l <= area_h; l++)
        for (int m = -area_w; m <= area_w; m++)
        {
            int out_w = (j - w_offset) / stride_w + m;
            int out_h = (i - h_offset) / stride_h + l;
            int out_index = out_w + w * (out_h + h * (k + c * b));
            int valid = (out_w >= 0 && out_w < w &&
                out_h >= 0 && out_h < h);
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
        }
    prev_delta[index] += d;
}

__global__ void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, MYTYPE* delta, MYTYPE* prev_delta, int* indexes)
{
    int h = (in_h + pad - size) / stride + 1;
    int w = (in_w + pad - size) / stride + 1;
    int c = in_c;
    int area = (size - 1) / stride;

    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;

    int index = id;
    int j = id % in_w;
    id /= in_w;
    int i = id % in_h;
    id /= in_h;
    int k = id % in_c;
    id /= in_c;
    int b = id;

    int w_offset = -pad / 2;
    int h_offset = -pad / 2;

    MYTYPE d = 0;
    int l, m;
    for (l = -area; l < area + 1; ++l) {
        for (m = -area; m < area + 1; ++m) {
            int out_w = (j - w_offset) / stride + m;
            int out_h = (i - h_offset) / stride + l;
            int out_index = out_w + w * (out_h + h * (k + c * b));
            int valid = (out_w >= 0 && out_w < w &&
                out_h >= 0 && out_h < h);
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
        }
    }
    prev_delta[index] += d;
}

void Maxpooling::Backward(Matrix& pre_layer_loss)
{
    loss(0, 0) = 1, loss(0, 1) = 6, loss(0, 2) = 7;
    loss(1, 0) = 9, loss(1, 1) = -5, loss(1, 2) = 44;
    loss(2, 0) = -77, loss(2, 1) = 2, loss(2, 2) = -10;
    loss.DataTransfer(HostToDevice);


    int n = input_cols * input_rows * units;

    UINT k = (n - 1) / BLOCK + 1;
    UINT x = k;
    UINT y = 1;
    if (x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK) + 1;
    }
    dim3 blocks = { x, y, 1 };
    pre_layer_loss.Zeroreset();
    backward_maxpool_layer_kernel <<<blocks, BLOCK>>> (n, input_rows, input_cols, units,
        stride_x, stride_y, kernel_rows, kernel_cols, 0,
        loss.GetDevMat(), pre_layer_loss.GetDevMat(), index);
    cudaDeviceSynchronize();
}

void Maxpooling::Save(std::string& _dir, int which)
{
    FILE* fp = fopen((_dir + "/" + "Maxpooling" + std::to_string(which) + ".txt").c_str(), "w");
    if (!fp)
    {
        printf("ERROR: open file %s failed.\n", (_dir + "/" + "Maxpooling" + std::to_string(which) + ".txt").c_str());
        getchar();
        return;
    }

    fprintf(fp, "loss:\n");
    loss.DataTransfer(DeviceToHost);
    for (int i = 0; i < loss.rows(); i++)
    {
        for (int j = 0; j < loss.cols(); j++)
            fprintf(fp, "%f ", loss(i, j));
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void Maxpooling::GetIndex(int* idx)
{
    if (!idx)
    {
        /*printf("ERROR: null pointer idx(int*) in Maxpooling::GetIndex\n");
        getchar();
        return;*/
        idx = new int[output.size()];
    }
    cudaMemcpy(idx, this->index, sizeof(int)*output.size(), cudaMemcpyDeviceToHost);
    for (int i = 0; i < output.size(); i++)
        printf("%d ", idx[i]);
}
