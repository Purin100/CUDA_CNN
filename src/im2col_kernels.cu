//This original file is from https://github.com/pjreddie/darknet/blob/master/src/im2col_kernels.cu
//some changes have been made for this project

#include "cuda_runtime.h"
#include "im2col.h"




// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void im2col_gpu_kernel(const int n, const MYTYPE* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        MYTYPE *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        MYTYPE* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const MYTYPE* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                //*data_col_ptr = data_im_ptr[ii * width + jj];

                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void im2col_gpu(MYTYPE *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, MYTYPE *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    //int height_col = (height + 2 * pad - ksize) / stride + 1;
    //int width_col = (width + 2 * pad - ksize) / stride + 1;
    int height_col = ((height + 2 * pad - ksize) / stride + 1) > 0 ? (height + 2 * pad - ksize) / stride + 1 : 1;
    int width_col = ((width + 2 * pad - ksize) / stride + 1) > 0 ? ((width + 2 * pad - ksize) / stride + 1) : 1;
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK, BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);
    cudaDeviceSynchronize();
}

void im2col_gpu(MYTYPE* im,
    int channels, int height, int width,
    int ksize, int stride, int pad, int height_col, int width_col, MYTYPE* data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel << <(num_kernels + BLOCK - 1) / BLOCK, BLOCK >> > (
        num_kernels, im, height, width, ksize, pad,
        stride, height_col,
        width_col, data_col);
    cudaDeviceSynchronize();
}

MYTYPE im2col_get_pixel(MYTYPE* im, int height, int width, int channels,
    int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(MYTYPE* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, MYTYPE* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}
