#pragma once
#include "utils.h"
#include "Matrix.cuh"
#include <math.h>
class Adam
{
public:
    bool Init(const int row, const int col, const char* dataType = nullptr, const MYTYPE lr = 1e-4);

    template <typename T>
    void Update(T& grad, T& weight);

    void SetpReset() { step = 0; }

    inline bool InitState() { return isInit; }
private:
    MYTYPE beta1 = 0.9f, beta2 = 0.999f;
    MYTYPE esp = 1e-6f;
    MYTYPE lr = 1e-4f;
    UINT step = 0;
    bool isInit = false;

    Matrix m, v, temp_loss;
    Matrix m_hat, v_hat;
    Vector m_v, v_v, temp_loss_v;
    Vector m_hat_v, v_hat_v;
};

template<typename T>
inline void Adam::Update(T& grad, T& weight)
{
    //assert((std::is_same<T, Vector>::value) || (std::is_same<T, Matrix>::value));

    step++;
    if (std::is_same<T, Vector>::value)
    {
        temp_loss_v = reinterpret_cast<Vector&>(grad);
        m_v = m_v * beta1 + (1.0 - beta1) * temp_loss_v;
        mc.VecEleMult(temp_loss_v.GetDevVec(), temp_loss_v.GetDevVec(), temp_loss_v.size());
        v_v = v_v * beta2 + (1.0 - beta2) * temp_loss_v;
        m_hat_v = m_v / (1.0 - pow(beta1, step));
        v_hat_v = v_v / (1.0 - pow(beta2, step));
        reinterpret_cast<Vector&>(weight) -= (lr * m_hat_v / (v_hat_v.vsqrt() + esp));
    }
    if(std::is_same<T,Matrix>::value)
    {
        temp_loss = reinterpret_cast<Matrix&>(grad);
        m = m * beta1 + (1.0 - beta1) * temp_loss;
        mc.MatrixEleMult(temp_loss.GetDevMat(), temp_loss.GetDevMat(), temp_loss.rows(), temp_loss.cols());
        v = v * beta2 + (1.0 - beta2) * temp_loss;
        m_hat = m / (1.0 - pow(beta1, step));
        v_hat = v / (1.0 - pow(beta2, step));
        v_hat.msqrt();
        mc.MatrixAddNum(v_hat.GetDevMat(), v.rows(), v.cols(), esp);
        mc.MatrixEleDiv(m_hat.GetDevMat(), v_hat.GetDevMat(), m_hat.rows(), m_hat.cols());
        reinterpret_cast<Matrix&>(weight) -= lr * m_hat;
    }
}
