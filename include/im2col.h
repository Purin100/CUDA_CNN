//This original file is from https://github.com/pjreddie/darknet/blob/master/src/im2col_kernels.cu
//some changes have been made for this project

#ifndef IM2COL_H
#define IM2COL_H
#include <device_launch_parameters.h>
#include "utils.h"
void im2col_cpu(MYTYPE* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, MYTYPE* data_col);

//#ifdef GPU

void im2col_gpu(MYTYPE *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,MYTYPE *data_col);

//#endif

void im2col_gpu(MYTYPE* im,
    int channels, int height, int width,
    int ksize, int stride, int pad, int height_col, int width_col, MYTYPE* data_col);
#endif
