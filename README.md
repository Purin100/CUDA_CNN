# CUDA_CNN
This project builds a CNN from scratch using C/C++. You need CUDA and OpenCV to run this project. If you find any bugs, please tell me. Thank you!

2D convolution layer, 2D max pooling layer, flatten layer, and fully connected layer (dense layer) are supported.

# Usage
1. Declear some layers like Conv2D c1, Flatten f, Dense d1, and the network Net net
2. Load the training data and test data
3. For the first layer, fill the relative LayerInfo structure. In this example, you need to declear and fill Conv2DInfo c1Info
4. Add the first layer using net.add(&c1, &c1Info)
5. Add other layers using function net.add
6. Use net.Forward function and net.Backward function to train the network with training data
7. Use net.Forward function and net.Eval function for testing process

The codes are tested on Windows 10 with MSVS 2019, CUDA 11.7.
