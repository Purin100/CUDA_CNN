#include "optimizer.h"

bool Adam::Init(const int row, const int col, const char* dataType, const MYTYPE lr)
{
    if (row <= 0 || col <= 0)
    {
        printf("ERROR: argument row <= 0 or(and) col <= 0. They should be larger than zero.\n");
        getchar();
        return false;
    }
    if (dataType == nullptr)
    {
        printf("ERROR: argument dataType is NULL. It should be vector or matrix.\n");
        getchar();
        return false;
    }
    if (lr > 0.0)
        this->lr = lr;

    if (mystrcmp(dataType, "vector") == 0)
    {
        assert(row == 1 || col == 1);
        m_v = Vector(row*col);
        v_v = Vector(row*col);
        memset(m_v.GetVec(), 0, sizeof(MYTYPE) * row*col);
        memset(v_v.GetVec(), 0, sizeof(MYTYPE) * row*col);
        cudaMemset(m_v.GetDevVec(), 0, sizeof(MYTYPE) * row*col);
        cudaMemset(v_v.GetDevVec(), 0, sizeof(MYTYPE) * row*col);
        isInit = true;
        return true;
    }
    if (mystrcmp(dataType, "matrix") == 0)
    {
        m = Matrix(row, col);
        v = Matrix(row, col);
        memset(m.GetMat(), 0, sizeof(MYTYPE) * row * col);
        memset(v.GetMat(), 0, sizeof(MYTYPE) * row * col);
        cudaMemset(m.GetDevMat(), 0, sizeof(MYTYPE) * row * col);
        cudaMemset(v.GetDevMat(), 0, sizeof(MYTYPE) * row * col);
        isInit = true;
        return true;
    }
    
    return false;
}
