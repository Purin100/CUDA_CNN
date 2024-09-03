//the original file is from https://github.com/pjreddie/darknet/blob/master/src/col2im.h
//some changes have been made for this project
#ifndef COL2IM_H
#define COL2IM_H

#include "utils.h"

//void col2im_cpu(float* data_col,
//    int channels, int height, int width,
//    int ksize, int stride, int pad, float* data_im);

//#ifdef GPU
//void col2im_gpu(MYTYPE* data_col,
//    int channels, int height/*rows in data_im*/, int width/*columns in data_im*/,
//    int ksize, int stride, int pad, int height_col, int width_col, MYTYPE* data_im);

void col2im_gpu(MYTYPE* data_col,
    int channels, int height/*rows in data_col*/, int width/*columns in data_col*/,
    int ksize, int stride, int pad, MYTYPE* data_im);

__global__ void col2im_gpu_kernel(const int n, const MYTYPE* data_col,
    const int height, const int width, const int ksize,
    const int pad,
    const int stride,
    const int height_col, const int width_col,
    MYTYPE* data_im);
//#endif
#endif