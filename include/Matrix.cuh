/*
Matrix and vector class are defined in this file.
These classes are used as basic parts in all kinds of layers in this project.
*/

#pragma once
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "utils.h"
#include "Mat_calc.cuh"


class Vector;
class Matrix;
//static mat_calc mc;//an object for matrix (and vector) calculation

//Generate an identity matrix sized in num * num.
Matrix Identity(const int num);

//Generate a matrix sized in row * col with all elements are zero
Matrix Zeros(const int row, const int col);

//Generate a matrix sized in row * col with all elements are one
Matrix Ones(const int row, const int col);

//In CUDA, a matrix is stored in column-mayjor. While in some CPU codes, a matrix is stroed in row-mayjor (like C++)
//If we need to copy matrix from GPU to CPU, and do something on CPU matrix, we need to change the matrix into row-mayjor.
//ATTENTION: src and dst should be two different pointers.
void Colmaj2Rowmaj(MYTYPE* src, MYTYPE* dst, int src_row, int src_col);

#define HostToDevice 1
#define DeviceToHost 2

enum ALLOCATETYPE
{
    NOTDEFINE = 0,
    CPU = 1,
    GPU = 2,
    CPUGPU = 3
};


class Vector
{
public:
    Vector() {};
    ~Vector() { if (data) delete[] data; if (dev_data) cudaFree(dev_data); };
    Vector(const int element_num);
    Vector(const Vector& v);
    Vector(MYTYPE* data, const int element_num);//Initialize a vector using an exsisting one dim array on CPU.

    MYTYPE* GetVec() { return data; }
    MYTYPE* GetDevVec() { return dev_data; }

    //copy a vector from CPU to GPU (HostToDevice), or from GPU to CPU (DeviceToHost)
    void DataTransfer(int trans_label);

    //free memory occupied by this vector (on CPU and on GPU) and reallocate a new space for this vector
    //all content in the vector will be cleaned
    //this function will do nothing if element_num == this->element_num
    void Realloc(int element_num);

    void showVector() { printf("Vector:"); for (int i = 0; i < element_num; i++) printf("%f ", data[i]); printf("\n"); }
    void ZeroReset();
    bool empty();
    int size() { return element_num; }

    //Assume a vector v=[v1, v2]
    // v.vsqrt() means v=[sqrt(v1), sqrt(v2)]
    Vector& vsqrt();

    //Calculate cos<this, v> as the angle between this vector and the given vector
    //return: a cos value in MYTYPE
    MYTYPE angle(const Vector& v);

    void Normalize();

    MYTYPE& operator[](int i);
    Vector& operator=(const Vector& v);

    Vector& operator-=(const Vector& v);
    Vector& operator-=(const MYTYPE num);
    Vector& operator+=(const Vector& v);
    Vector& operator*=(const MYTYPE num);
    Vector& operator/=(const MYTYPE num);

    bool operator==(const Vector& v);

    friend Vector operator*(const MYTYPE num, const Vector& v);
    friend Vector operator*(const Vector& v, const MYTYPE num);
    friend Vector operator-(const Vector& v, const MYTYPE num);
    friend Vector operator-(const Vector& vec_a, const Vector& vec_b);
    friend Vector operator/(const Vector& v, const MYTYPE num);
    friend Vector operator/(const Vector& vec_a, const Vector& vec_b);
    friend Vector operator+(const Vector& vec_a, const Vector& vec_b);
    friend Vector operator+(const Vector& v, const MYTYPE num);

    
private:
    MYTYPE* data = nullptr;//data on CPU
    MYTYPE* dev_data = nullptr;//data on GPU
    int element_num = 0;//how many numbers in this vector
protected:
    void _init(const int element_num);
};



class Matrix
{
public:
    Matrix() {};
    Matrix(int row, int col, ALLOCATETYPE type = CPUGPU);
    Matrix(const Matrix& m, ALLOCATETYPE type = NOTDEFINE);
    ~Matrix();

    inline int rows() { return row; }
    inline int cols() { return col; }
    inline int size() { return element_num; }
    MYTYPE* GetMat() { return data; }
    MYTYPE* GetDevMat() { return dev_data; }
    Vector RowSlice(const int which_row);//Get a row in the matrix (in Vector type)

    //Fill a row in the matrix with given vector vec if which >= 0 && which < this->row.
    //If which < 0 or which > this->row, then fill all rows in the matrix with given vector vec.
    void RowFill(Vector vec, const int which = -1);

    void Zeroreset();//set all elements in the matrix to zero

    void reshape(const int dst_row, const int dst_col, const int dst_channels);

    //copy a matrix from CPU to GPU (HostToDevice), or from GPU to CPU (DeviceToHost)
    void DataTransfer(int trans_label);
    void showMat();//print data on CPU

    //Element-wise division another matrix
    void EleDiv(const Matrix& mat);

    //Assume a matrix m=([m1, m2],
    //                  [m3, m4])
    // m.msqrt() means m=([sqrt(m1), sqrt(m2)],
    //                   [sqrt(m3), sqrt(m4)])
    //This function won't process values less than zero, it will set these values to zero.
    //For example, one of the values in the matrix M called m1 = -1.0, then after calling M.msqrt()
    // m1 = 0.0
    Matrix& msqrt();

    bool empty();

    MYTYPE& operator()(int i, int j);//you can gain access to an element (on CPU memory) at i-th row and j-th column
    Matrix& operator=(const Matrix& m);

    friend Matrix operator-(const Matrix& m, const Matrix& n);
    friend Matrix operator+(const Matrix& m, const Matrix& n);
    friend Matrix operator*(const Matrix& mat, const MYTYPE num);
    friend Matrix operator*(const MYTYPE num, const Matrix& mat);
    friend Matrix operator/(const Matrix& mat, const MYTYPE num);
    friend Matrix operator-(const Matrix& mat, const MYTYPE num);
    friend Matrix operator-(const MYTYPE num, const Matrix& mat);

    Matrix& operator-=(const Matrix& m);
    Matrix& operator+=(const Matrix& m);
    Matrix& operator+=(const MYTYPE num);
    Matrix& operator*=(const MYTYPE num);
    Matrix& operator/=(const MYTYPE num);
    
private:
    //In CUDA, matrix is stored in an one-dim array,
    //so I use an one-dim pointer
    MYTYPE* data = nullptr;//data on CPU
    MYTYPE* dev_data = nullptr;//data on GPU
    int row = 0, col = 0;
    int element_num = 0;//how many numbers in this matrix
protected:
    void _init(const int row, const int col, ALLOCATETYPE type = CPUGPU);
};