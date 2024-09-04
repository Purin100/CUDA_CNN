#include "Matrix.cuh"
#include "Mat_calc.cuh"

//Vector part
Vector::Vector(const int element_num)
{
    _init(element_num);
}

Vector::Vector(const Vector& v)
{
    assert(v.element_num > 0);
    {
        if (element_num == 0)
            _init(v.element_num);
        if (v.data)
            memcpy(this->data, v.data, sizeof(MYTYPE) * this->element_num);
        if (v.dev_data)
            cudaMemcpy(dev_data, v.dev_data, sizeof(MYTYPE) * this->element_num, cudaMemcpyDeviceToDevice);
    }
}

Vector::Vector(MYTYPE* data, const int element_num)
{
    if (element_num > 0 && data)
    {
        _init(element_num);
        memcpy(this->data, data, sizeof(MYTYPE) * this->element_num);
        cudaMemcpy(dev_data, data, sizeof(MYTYPE) * this->element_num, cudaMemcpyHostToDevice);
    }

}

void Vector::DataTransfer(int trans_label)
{
    if (trans_label == HostToDevice)
        cudaMemcpy(dev_data, data, sizeof(MYTYPE) * element_num, cudaMemcpyHostToDevice);
    if (trans_label == DeviceToHost)
        cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
}

void Vector::Realloc(int element_num)
{
    if (element_num == this->element_num)
        return;
    if (data)
    {
        delete[] data;
        data = nullptr;
    }
    cudaFree(dev_data);
    _init(element_num);
}

void Vector::_init(const int element_num)
{
    if (element_num > 0)
    {
        this->element_num = element_num;
        data = new MYTYPE[element_num];
        cudaMalloc((void**)&dev_data, sizeof(MYTYPE) * element_num);
    }
}

void Vector::ZeroReset()
{
    if (data)
        memset(data, 0, sizeof(MYTYPE) * element_num);
    if (dev_data)
        cudaMemset(dev_data, 0, sizeof(MYTYPE) * element_num);
}

bool Vector::empty()
{
    return element_num == 0 || data == nullptr;
}

Vector& Vector::vsqrt()
{
    mc.VecSqrt(this->dev_data, this->element_num);
    cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
    return *this;
}

MYTYPE Vector::angle(const Vector& v)
{
    assert(this->element_num == v.element_num);
    MYTYPE result;

    result = mc.VecAngle(this->dev_data, v.dev_data, element_num);

    return result;
}

void Vector::Normalize()
{
    mc.VecNormalize(dev_data, element_num);
    cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
}

MYTYPE& Vector::operator[](int i)
{
    assert(data && i < this->element_num && i >= 0);
    return this->data[i];
}

Vector operator+(const Vector& vec_a, const Vector& vec_b)
{
    assert(vec_a.element_num == vec_b.element_num);
    Vector t = vec_a;
    mc.VectorAdd(t.dev_data, vec_a.dev_data, vec_b.dev_data, vec_a.element_num);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Vector operator-(const Vector& v, const MYTYPE num)
{
    Vector t = v;
    mc.VecSubNum(t.dev_data, num, t.element_num);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Vector operator+(const Vector& v, const MYTYPE num)
{
    Vector t = v;
    mc.VecAddNum(t.dev_data, num, t.element_num);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Vector& Vector::operator+=(const Vector& v)
{
    assert(v.element_num == this->element_num);
    mc.VectorAdd(this->dev_data, v.dev_data, this->dev_data, element_num);
    cudaMemcpy(data, dev_data, element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    return *this;
}

Vector& Vector::operator=(const Vector& v)
{
    assert(v.element_num > 0);
    if (this != &v)
    {
        if (this->element_num > 0)
        {
            assert(this->element_num == v.element_num);
            memcpy(this->data, v.data, sizeof(MYTYPE) * element_num);
            cudaMemcpy(dev_data, v.dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToDevice);
        }
        else
        {
            this->element_num = v.element_num;
            this->_init(v.element_num);
            memcpy(this->data, v.data, sizeof(MYTYPE) * element_num);
            cudaMemcpy(dev_data, v.dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToDevice);
        }
        
    }
    return *this;
}

Vector operator-(const Vector& vec_a, const Vector& vec_b)
{
    assert(vec_a.element_num == vec_b.element_num);
    Vector t = vec_a;
    mc.VecSub(vec_a.dev_data, vec_b.dev_data, t.dev_data, t.element_num);
    cudaMemcpy(t.data, t.dev_data, t.element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    return t;
}

Vector operator/(const Vector& v, const MYTYPE num)
{
    Vector t;
    t = v;
    mc.VecDivNum(t.dev_data, num, t.element_num);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Vector operator/(const Vector& vec_a, const Vector& vec_b)
{
    assert(vec_a.element_num == vec_b.element_num);
    Vector t = vec_a;
    mc.VecEleDiv(t.dev_data, vec_b.dev_data, t.element_num);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Vector& Vector::operator-=(const Vector& v)
{
    assert(this->element_num == v.element_num);
    mc.VecSub(dev_data, v.dev_data, dev_data, element_num);
    cudaMemcpy(data, dev_data, element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    return *this;
}

Vector& Vector::operator-=(const MYTYPE num)
{
    mc.VecSubNum(dev_data, num, element_num);
    cudaMemcpy(data, dev_data, element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    return *this;
}

Vector operator*(const Vector& v, const MYTYPE num)
{
    Vector t;
    t = v;
    assert(!t.empty());
    mc.VecMultNum(t.dev_data, num, t.element_num);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Vector operator*(const MYTYPE num, const Vector& v)
{
    Vector t;
    t = v;
    assert(!t.empty());
    mc.VecMultNum(t.dev_data, num, t.element_num);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Vector& Vector::operator*=(const MYTYPE num)
{
    if (this->element_num > 0)
    {
        mc.VecMultNum(this->dev_data, num, this->element_num);
        cudaMemcpy(data, dev_data, element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    }
    return *this;
}

Vector& Vector::operator/=(const MYTYPE num)
{
    assert(num != 0.0 && this->element_num > 0);
    mc.VecDivNum(dev_data, num, element_num);
    cudaMemcpy(data, dev_data, element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    return *this;
}

bool Vector::operator==(const Vector& v)
{
    if (this->element_num != v.element_num)
        return false;

    //this->DataTransfer(DeviceToHost);
    //cudaMemcpy(v.data, v.dev_data, sizeof(MYTYPE) * v.element_num, cudaMemcpyDeviceToHost);
    //int threads = 32, blocks = (element_num - threads + 1) / threads;
    for (int i = 0; i < element_num; i++)
        if (this->data[i] != v.data[i])
            return false;

    return true;
}


//Matrix part
Matrix::Matrix(int row, int col, ALLOCATETYPE type)
{
    _init(row, col, type);
}

void Matrix::_init(int row, int col, ALLOCATETYPE type)
{
    assert(row > 0 && col > 0);
    assert(type != NOTDEFINE);
    this->row = row, this->col = col;
    element_num = row * col;
    if (type == CPU)
        data = new MYTYPE[element_num];
    if (type == GPU)
        cudaMalloc((void**)&dev_data, sizeof(MYTYPE) * element_num);
    if (type == CPUGPU)
    {
        data = new MYTYPE[element_num];
        cudaMalloc((void**)&dev_data, sizeof(MYTYPE) * element_num);
    }
}

Matrix::Matrix(const Matrix& m, ALLOCATETYPE type)
{
    if (type == NOTDEFINE)
    {
        if (m.data && m.dev_data)
            type = CPUGPU;
        if (m.data && !m.dev_data)
            type = CPU;
        if (!m.data && m.dev_data)
            type = GPU;
    }
    _init(m.row, m.col, type);
    if (data)
        memcpy(data, m.data, sizeof(MYTYPE) * element_num);
    if (dev_data)
        cudaMemcpy(dev_data, m.dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToDevice);
}

Matrix::~Matrix()
{
    if (data)
    {
        delete[] data;
        data = nullptr;
    }
    if (dev_data)
        cudaFree(dev_data);
}

MYTYPE& Matrix::operator()(int i, int j)
{
    assert(i >= 0 && i < row && j >= 0 && j < col && data);
    return data[i * col + j];
}

Matrix& Matrix::operator=(const Matrix& m)
{
    if (this != &m)
    {
        if (this->element_num != 0)
            assert(m.row == this->row && m.col == this->col);
        else
            this->_init(m.row, m.col);

        if(data)
            memcpy(data, m.data, sizeof(MYTYPE) * element_num);
        if(dev_data)
            cudaMemcpy(dev_data, m.dev_data, sizeof(MYTYPE)*element_num, cudaMemcpyDeviceToDevice);
    }
    return *this;
}

Matrix operator-(const Matrix& m, const Matrix& n)
{
    assert(n.col == m.col && n.row == m.row);
    Matrix t;
    t = m;
    mc.MatrixSub(t.dev_data, n.dev_data, m.row, m.col);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Matrix operator+(const Matrix& m, const Matrix& n)
{
    assert(n.col == m.col && n.row == m.row);
    Matrix t = m;
    mc.MatrixAdd(t.dev_data, n.dev_data, m.row, m.col);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Matrix& Matrix::operator-=(const Matrix& m)
{
    assert(this->col == m.col && this->row == m.row);
    mc.MatrixSub(this->dev_data, m.dev_data, row, col);
    cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
    return *this;
}

Matrix& Matrix::operator+=(const Matrix& m)
{
    assert(this->row == m.row && this->col == m.col);
    mc.MatrixAdd(this->dev_data, m.dev_data, row, col);
    cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
    return *this;
}

Matrix& Matrix::operator+=(const MYTYPE num)
{
    mc.MatrixAddNum(dev_data, row, col, num);
    cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
    return *this;
}

Matrix& Matrix::operator*=(const MYTYPE num)
{
    assert(this->element_num > 0);
    mc.MatrixMultNumber(dev_data, num, row, col);
    cudaMemcpy(data, dev_data, sizeof(MYTYPE) * row * col, cudaMemcpyDeviceToHost);
    return *this;
}

Matrix& Matrix::operator/=(const MYTYPE num)
{
    assert(num != 0);
    mc.MatrixDivNumber(dev_data, num, row, col);
    cudaMemcpy(data, dev_data, sizeof(MYTYPE) * row * col, cudaMemcpyDeviceToHost);
    return *this;
}

Matrix operator*(const Matrix& mat, const MYTYPE num)
{
    Matrix t;
    t = mat;
    mc.MatrixMultNumber(t.dev_data, num, t.row, t.col);
    t.DataTransfer(DeviceToHost);
    return t;
}

Matrix operator*(const MYTYPE num, const Matrix& mat)
{
    Matrix t = mat;
    mc.MatrixMultNumber(t.dev_data, num, t.row, t.col);
    t.DataTransfer(DeviceToHost);
    return t;
}

Matrix operator/(const Matrix& mat, const MYTYPE num)
{
    assert(num != 0);
    Matrix t = mat;
    mc.MatrixMultNumber(t.dev_data, 1.0 / num, t.row, t.col);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Matrix operator-(const Matrix& mat, const MYTYPE num)
{
    Matrix t = mat;
    mc.MatrixAddNum(t.dev_data, t.row, t.col, -num);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Matrix operator-(const MYTYPE num, const Matrix& mat)
{
    Matrix t = mat;
    mc.NumSubMat(t.dev_data, num, t.row, t.col);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Vector Matrix::RowSlice(const int which_row)
{
    assert(which_row >= 0 && which_row < row);
    Vector v(col);
    for (int i = 0; i < col; i++)
        v[i] = data[which_row * col + i];
    v.DataTransfer(HostToDevice);
    return v;
}

void Matrix::DataTransfer(int trans_label)
{
    if (trans_label == HostToDevice)
        cudaMemcpy(dev_data, data, sizeof(MYTYPE) * element_num, cudaMemcpyHostToDevice);
    if (trans_label == DeviceToHost)
        cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
}

void Matrix::showMat()
{
    for (int i = 0; i < row; i++)
    {
        printf("row %d:\n", i);
        for (int j = 0; j < col; j++)
            printf("%f ", data[i * col + j]);
        printf("\n");
    }
}

void Matrix::EleDiv(const Matrix& mat)
{
    assert(row == mat.row && col == mat.col);
    mc.MatrixEleDiv(dev_data, mat.dev_data, row, col);
    cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
}

Matrix& Matrix::msqrt()
{
    mc.MatSqrt(dev_data, row, col);
    cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
    return *this;
}

bool Matrix::empty()
{
    if (element_num <= 0 || !data || !dev_data)
        return true;
    else
        return false;
}

void Matrix::RowFill(Vector vec, const int which)
{
    assert(!vec.empty());
    assert(vec.size() == col);
    if (which >= 0 && which < row)
    {
        memcpy(&data[col * which], vec.GetVec(), sizeof(MYTYPE) * col);
    }
    else
    {
        for (int i = 0; i < element_num; i += col)
            memcpy(&data[i], vec.GetVec(), sizeof(MYTYPE) * col);
    }
    cudaMemcpy(dev_data, data, sizeof(MYTYPE) * element_num, cudaMemcpyHostToDevice);
}

void Matrix::Zeroreset()
{
    assert(data != nullptr && dev_data != nullptr);
    memset(data, 0, sizeof(MYTYPE) * element_num);
    cudaMemset(dev_data, 0, sizeof(MYTYPE) * element_num);
}

// CUDA kernel for reshaping matrix
__global__ void Mat_reshape_kernel(MYTYPE* input, MYTYPE* output, const int row, const int col, const int channels) {
    /*int channels = 3;
    int height = 26;
    int width = 26;*/

    int idx_out = threadIdx.x + blockIdx.x * blockDim.x;  // Global index in output array

    if (idx_out < channels * row * col) {
        int c = idx_out / (row * col);   // Calculate channel index
        int offset = idx_out % (row * col);  // Calculate offset within the channel

        // Calculate indices in input and output arrays
        int idx_in = c * row * col + offset;  // Index in input array
        output[idx_out] = input[idx_in];  // Perform the reshape operation
    }
}
void Matrix::reshape(const int dst_row, const int dst_col, const int dst_channels)
{
    assert(this->element_num == dst_row * dst_col * dst_channels);

    int threads = 32;
    int blocks = (element_num + threads - 1) / threads;
    Mat_reshape_kernel <<<blocks, threads>>> (dev_data, dev_data, dst_row, dst_col, dst_channels);
    cudaDeviceSynchronize();

    row = dst_row;
    col = dst_col;
}

Matrix Identity(const int num)
{
    assert(num > 0);
    Matrix mat = Zeros(num, num);
    for (int i = 0; i < num; i++)
        mat(i, i) = 1.0;
    mat.DataTransfer(HostToDevice);
    return mat;
}
Matrix Zeros(const int row, const int col)
{
    assert(row > 0 && col > 0);
    Matrix mat(row, col);
    //for (int i = 0; i < row; i++)
    //    for (int j = 0; j < col; j++)
    //        mat(i, j) = 0.0;
    memset(mat.GetMat(), 0, sizeof(MYTYPE) * row * col);
    cudaMemset(mat.GetDevMat(), 0, sizeof(MYTYPE) * row * col);
    //mat.DataTransfer(HostToDevice);
    return mat;
}

Matrix Ones(const int row, const int col)
{
    Matrix mat(row, col);
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            mat(i, j) = 1.0;
    mat.DataTransfer(HostToDevice);
    return mat;
}

void Colmaj2Rowmaj(MYTYPE* src, MYTYPE* dst, int src_row, int src_col)
{
    if (src == nullptr || dst == nullptr)
    {
        printf("ERROR: src or dst in function Colmaj2Rowmaj(MYTYPE*, MYTYPE*) is null pointer.\n");
        getchar();
        return;
    }

    int idx_src = 0, idx_dst = 0;
    for(; idx_src < src_row; idx_src++)
        for (int i = 0; i < src_col; i++)
        {
            dst[idx_dst] = src[idx_src + src_row * i];
            idx_dst++;
        }
}
