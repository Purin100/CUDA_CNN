/*
This file difines Flatten layer.
*/
#ifndef __FLATTEN_H__
#define __FLATTEN_H__

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include "Matrix.cuh"
#include "Conv2D.cuh"
#include "Pooling.cuh"

using std::vector;

class Flatten
{
public:
    Flatten();
    ~Flatten();

    bool BuildLayer(const int input_row, const int input_col);

    void Forward(Conv2D* _input, int num = 1);
    void Forward(MYTYPE** _input, int row, int col);
    void Forward(Vector& _input);
    void Forward(Maxpooling* _input, int num = 1);

    void Backward(Vector& _loss);
    void Backward(Conv2D* pre_layer);
    void Backward(pMaxp pre_layer);

    void Reverse(Vector reverse_vec);

    void DisplayOutput();
    void Save(const char* dir_name, const char* mode);
    //float* GetOutput() {return output;}
    Vector GetOutput() {return output;}
    //int GetSize(){return size;}
    int GetSize() { return output.size();}

    Vector loss;
    Vector output;
    Vector reverse_output;
private:
    
    
    int size;
};

#endif