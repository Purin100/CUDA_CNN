//The original file is from https://github.com/pjreddie/darknet/blob/master/src/col2im_kernels.cu
//some changes have been made for this project
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include "cublas_v2.h"
#include "col2im.h"

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void col2im_gpu_kernel(const int n, const MYTYPE* data_col,
    const int height, const int width, const int ksize,
    const int pad,
    const int stride,
    const int height_col, const int width_col,
    MYTYPE* data_im) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < n; index += blockDim.x * gridDim.x) {
        MYTYPE val = 0;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);
        // compute the start and end of the output
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        // equivalent implementation
        int offset =
            (c * ksize * ksize + h * ksize + w) * height_col * width_col;
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        int coeff_w_col = (1 - stride * height_col * width_col);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] += val;
    }
}

void col2im_gpu(MYTYPE* data_col,
    int channels, int height/*rows in data_im*/, int width/*columns in data_im*/,
    int ksize, int stride, int pad, MYTYPE* data_im) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = ((height + 2 * pad - ksize) / stride + 1) > 0 ? (height + 2 * pad - ksize) / stride + 1 : 1;
    int width_col = ((width + 2 * pad - ksize) / stride + 1) > 0 ? ((width + 2 * pad - ksize) / stride + 1) : 1;
    //width_col = (width - 1) * stride + ksize - 2 * pad;
    //height_col = (height - 1) * stride + ksize - 2 * pad;

    //int num_kernels = channels * height_col * width_col;
    int num_kernels = channels * height * width;
    col2im_gpu_kernel <<<(num_kernels + BLOCK - 1) / BLOCK,
        BLOCK >>> (
            num_kernels, data_col, height, width, ksize, pad,
            stride, height_col,
            width_col, data_im);
    cudaDeviceSynchronize();
}

//void col2im_gpu(MYTYPE* data_col,
//    int channels, int height/*rows in data_im*/, int width/*columns in data_im*/,
//    int ksize, int stride, int pad, int height_col, int width_col, MYTYPE* data_im)
//{
//    int num_kernels = channels * height_col * width_col;
//    col2im_gpu_kernel <<<(num_kernels + BLOCK - 1) / BLOCK,
//        BLOCK >>> (
//            num_kernels, data_col, height, width, ksize, pad,
//            stride, height_col,
//            width_col, data_im);
//    cudaDeviceSynchronize();
//}