#include "Conv2D.cuh"

static std::random_device conv_rd;
static std::normal_distribution<MYTYPE> uni(0.0,0.01);

Conv2D::Conv2D()
{
    input_row=0, input_col=0, output_row=0, output_col=0;
    padding_rows = 0;
    padding_cols = 0;
    channels = 0;
    stride_rows = 0, stride_cols = 0;
    padding_mode = VALID;
}

bool Conv2D::BuilderLayer(int input_row, int input_col, int k_row, int k_col,
    int stride_row, int stride_col, int channels, int units, int input_units,
    const char* padding_mode, const char* activation, const char* optimizer)
{
    if (input_row <= 0 || input_col <= 0)
    {
        printf("ERROR: invalid argument(s) input_row and/or input_col. Please ensure input_row > 0 and input_col > 0.\n");
        getchar();
        return false;
    }
    if (k_row <= 0 || k_col <= 0)
    {
        printf("ERROR: invalid argument(s) k_row and/or k_col. Please ensure k_row > 0 and k_col > 0.\n");
        getchar();
        return false;
    }
    if (stride_row <= 0 || stride_col <= 0)
    {
        printf("ERROR: invalid argument(s) stride_row and/or stride_col. Please ensure stride_row > 0 and stride_col > 0.\n");
        getchar();
        return false;
    }
    if (channels <= 0)
    {
        printf("ERROR: invalid argument channels. Please ensure channels > 0.\n");
        getchar();
        return false;
    }
    if (units <= 0)
    {
        printf("ERROR: invalid argument channels. Please ensure units > 0.\n");
        getchar();
        return false;
    }
    if (activation == nullptr)
    {
        printf("ERROR: invalid argument activation (null pointer).\n");
        getchar();
        return false;
    }

    this->input_row = input_row, this->input_col = input_col;
    this->kernel_row = k_row, this->kernel_col = k_col;
    this->stride_rows = stride_row, this->stride_cols = stride_col;

    if (mystrcmp(activation,"relu")==0)
    {
        this->activation = &Relu;
        gradient = &Relu_gradient;
        goto ENDACCONV;
    }
    if (mystrcmp(activation, "relu1") == 0)
    {
        this->activation = &Relu1;
        this->gradient = &Relu1_gradient;
        goto ENDACCONV;
    }
    if (mystrcmp(activation, "tanh") == 0)
    {
        this->activation = &Tanh;
        gradient = &Tanh_gradient;
        goto ENDACCONV;
    }
    if (mystrcmp(activation, "linear") == 0)
    {
        this->activation = &Linear;
        gradient = &Linear_gradient;
        goto ENDACCONV;
    }
    if (mystrcmp(activation, "sigmoid") == 0)
    {
        this->activation = &Sigmoid;
        gradient = &Sigmoid_gradient;
        goto ENDACCONV;
    }
    if (this->activation == nullptr)
    {
        this->activation = &Tanh;
        this->gradient = &Tanh_gradient;
    }
ENDACCONV:
    this->channels = channels;
    this->units = units;
    this->input_units = input_units;
    kernel_units = units * k_row * k_col;

    //kernel im2col
    //in the kernel matrix this->kernel, each row contains all elements in one convolutional kernel
    //the number of rows indicate how many kernels in this convolutional layer
    //e.g. if you have 4 kernels in this layer, and each kernel sizes in 2*2, and input channel is one,
    //then you need to initialize this->kernel = Matrix(4, 2*2*1)
    //a 2*2 kernel 
    // | k00, k01 |
    // | k10, k11 |
    //will be transfered into a row vector
    // (k00, k01, k10, k11)
    kernel = Matrix(units, k_row * k_col * channels);
    for (int i = 0; i < units; i++)
        for (int j = 0; j < k_row * k_col * channels; j++)
        {
            kernel(i, j) = uni(conv_rd);
        }

    kernel.DataTransfer(HostToDevice);
    kernel_grad = Matrix(units, k_row * k_col * channels);
    save_grad = Matrix(units, k_row * k_col * channels);
    grad_sample = Matrix(units, k_row * k_col * channels);
    kernel_t = Matrix(k_row * k_col * channels, units);
    if (input_units == 1)
        input = Matrix(input_row, input_col);
    else
        input = Matrix(input_units, input_row * input_col);

    if (mystrcmp(padding_mode, "same") == 0)
    {
        padding_rows = (kernel_row - 1) / 2;
        padding_cols = (kernel_col - 1) / 2;
    }
    else
    {
        padding_rows = 0, padding_cols = 0;
    }

    output_row = (input_row + 2 * padding_rows - kernel_row) / stride_rows + 1;
    output_col = (input_col + 2 * padding_cols - kernel_col) / stride_cols + 1;
    if (output_row == 0 && output_col == 0)
    {
        printf("ERROR: invalid output shape (output_row=%d, output_col=%d) in Conv2D::Buildlayer.\n Please check the values in input_row and input_col.\n", output_row, output_col);
        getchar();
        return false;
    }
    if (output_row == 0)
        output_row = 1;
    if (output_col == 0)
        output_col = 1;
    output = Matrix(units, output_row * output_col);

    local_out = Matrix(units, output_row * output_col);
    differial = Matrix(units, output_row * output_col);

    loss = Matrix(units, output_row * output_col);
    kernel_grad.Zeroreset();
    if ((optimizer) && mystrcmp(optimizer, "adam") == 0)
    {
        adam = new Adam;
        if (!adam->Init(loss.rows(), loss.cols(), "matrix"))
            return false;
    }
    
    return true;
}

void Conv2D::Forward(Conv2D* pre_layer)
{
    if (!pre_layer)
    {
        printf("ERROR: parament pre_layer in Conv2D layer is null pointer.\n");
        getchar();
        return;
    }

    int threads = 64;
    int blocks = (local_out.size() + threads - 1) / threads;

    input = pre_layer->output;
    _im2col(pre_layer->output, colImg, input_row, input_col,
        kernel_row, kernel_col, pre_layer->units, stride_rows, stride_cols, padding_rows, padding_cols);
   
    colImg.DataTransfer(DeviceToHost);
    mc.gemm_gpu(kernel.GetDevMat(), kernel.rows(), kernel.cols(), colImg.GetDevMat(), colImg.rows(), colImg.cols(),
        local_out.GetDevMat(), local_out.rows(), CUBLAS_OP_N, CUBLAS_OP_N);
    activation <<<blocks, threads>>> (local_out.GetDevMat(), output.GetDevMat(), output.size());
    cudaDeviceSynchronize();
    output.DataTransfer(DeviceToHost);
}

void Conv2D::Forward(pMaxp pre_layer)
{
    if (!pre_layer)
    {
        printf("ERROR: parament pre_layer (Maxpooling*) in Conv2D layer is null pointer.\n");
        getchar();
        return;
    }

    int threads = 64;
    int blocks = (pre_layer->size() + threads - 1) / threads;

    _im2col(pre_layer->Getoutput(), colImg, pre_layer->OutRow(), pre_layer->OutCol(), kernel_row, kernel_col, 
        pre_layer->unit(), stride_rows, stride_cols, padding_rows, padding_cols);
    colImg.DataTransfer(DeviceToHost);
    /*mc.gemm_gpu(kernel.GetDevMat(), kernel.rows(), kernel.cols(), colImg.GetDevMat(), colImg.rows(), colImg.cols(), 
        output.GetDevMat(), output.rows(), CUBLAS_OP_N, CUBLAS_OP_N);*/
    mc.gemm_gpu(kernel.GetDevMat(), kernel.rows(), kernel.cols(), colImg.GetDevMat(), colImg.rows(), colImg.cols(),
        local_out.GetDevMat(), local_out.rows(), CUBLAS_OP_N, CUBLAS_OP_N);
    activation <<<blocks, threads>>> (local_out.GetDevMat(), output.GetDevMat(), output.size());
    cudaDeviceSynchronize();
    output.DataTransfer(DeviceToHost);
}

void Conv2D::Forward(MYTYPE** _input, int row, int col)
{
    assert(row == input_row && col == input_col && _input);
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            input(i, j) = _input[i][j];
    input.DataTransfer(HostToDevice);
    //_im2col(input, colImg, row, col, kernel_row, kernel_col, channels, stride_rows, stride_cols, padding_rows, padding_cols);

    dim3 threads(32, 32, 1);
    dim3 blocks((output_row + threads.x - 1) / threads.x, (output_col + threads.y - 1) / threads.y, 1);

    _im2col(input, colImg, row, col, kernel_row, kernel_col, channels, stride_rows, stride_cols, padding_rows, padding_cols);
    if(units==1)
        mc.gemm_gpu(kernel.GetDevMat(), kernel.rows(), kernel.cols(), colImg.GetDevMat(), colImg.rows(), colImg.cols(), 
            local_out.GetDevMat(), 1, CUBLAS_OP_N, CUBLAS_OP_N);
    else
        mc.gemm_gpu(kernel.GetDevMat(), kernel.rows(), kernel.cols(), colImg.GetDevMat(), colImg.rows(), colImg.cols(), 
            local_out.GetDevMat(), local_out.rows(), CUBLAS_OP_N, CUBLAS_OP_N);
    activation <<<blocks, threads>>> (local_out.GetDevMat(), output.GetDevMat(), output.size());
    cudaDeviceSynchronize();
    output.DataTransfer(DeviceToHost);
}

void Conv2D::Forward(Vector& _input, int row, int col, int channels)
{
    if (_input.empty())
    {
        printf("ERROR: _input (Vector&) is empty in Conv2D::Forward(Vector&, int, int, int)\n");
        getchar();
        return;
    }
    assert(row > 0 && col > 0 && channels > 0);

    int count = 0;
    for(int i=0;i<input.rows();i++)
        for (int j = 0; j < input.cols(); j++)
        {
            input(i, j) = _input[count];
            count++;
        }
    input.DataTransfer(HostToDevice);

    int threads = 64;
    int blocks = (local_out.size() + threads - 1) / threads;

    _im2col(input, colImg, row, col, kernel_row, kernel_col, channels, stride_rows, stride_cols, padding_rows, padding_cols);
    mc.gemm_gpu(kernel.GetDevMat(), kernel.rows(), kernel.cols(), colImg.GetDevMat(), colImg.rows(), colImg.cols(),
        local_out.GetDevMat(), local_out.rows(), CUBLAS_OP_N, CUBLAS_OP_N);

    activation <<<blocks, threads>>> (local_out.GetDevMat(), output.GetDevMat(), output.size());
    cudaDeviceSynchronize();
    output.DataTransfer(DeviceToHost);
}

void Conv2D::Forward(MYTYPE* _input, int row, int col)
{
    assert(row == input_row && col == input_col && _input);
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            input(i, j) = _input[i * col + j];
    input.DataTransfer(HostToDevice);

    _im2col(input, colImg, row, col, kernel_row, kernel_col, channels, stride_rows, stride_cols, padding_rows, padding_cols);
    dim3 threads(32, 32, 1);
    dim3 blocks((output_row + threads.x - 1) / threads.x, (output_col + threads.y - 1) / threads.y, 1);
    if (units == 1)
        mc.gemm_gpu(kernel.GetDevMat(), kernel.rows(), kernel.cols(), colImg.GetDevMat(), colImg.rows(), colImg.cols(), 
            local_out.GetDevMat(), 1, CUBLAS_OP_N, CUBLAS_OP_N);
    else
        mc.gemm_gpu(kernel.GetDevMat(), kernel.rows(), kernel.cols(), colImg.GetDevMat(), colImg.rows(), colImg.cols(), 
            local_out.GetDevMat(), local_out.rows(), CUBLAS_OP_N, CUBLAS_OP_N);
    activation <<<blocks, threads>>> (local_out.GetDevMat(), output.GetDevMat(), output.size());
    cudaDeviceSynchronize();
    output.DataTransfer(DeviceToHost);
}

void Conv2D::_im2col(Matrix& src, Matrix& dst, int src_row, int src_col, int kernel_row, int kernel_col, int kernel_units/*channels*/,
    int stride_row, int stride_col, int padding_row, int padding_col)
{
    assert(kernel_row > 0 && kernel_col > 0 && kernel_units > 0);
    int colImg_row, colImg_col;//rows and columns for images after im2col

    if (dst.rows() == 0 && dst.cols() == 0)
    {
        assert(src_row > 0 && src_col > 0);
        assert(src_row > 1 || src_col > 1);
        adj_row = (src_row + 2 * padding_row - kernel_row) % stride_row;
        adj_col = (src_col + 2 * padding_col - kernel_col) % stride_col;
        if (src_row == 1)
            colImg_col = (src_col + 2 * padding_col - kernel_col) / stride_col + 1;
        if (src_col == 1)
            colImg_col = (src_row + 2 * padding_row - kernel_row) / stride_row + 1;
        if (src_row > 1 && src_col > 1)
            colImg_col = ((src_row + 2 * padding_row - kernel_row) / stride_row + 1) * ((src_col + 2 * padding_col - kernel_col) / stride_col + 1);
        colImg_row = kernel_row * kernel_col * kernel_units;
        dst = Matrix(colImg_row, colImg_col);
    }
    im2col_gpu(src.GetDevMat(), kernel_units, src_row, src_col, kernel_row, stride_row, padding_row, dst.GetDevMat());
    dst.DataTransfer(DeviceToHost);
}

void Conv2D::_col2im(Matrix& src, Matrix& dst, int src_row, int src_col, int kernel_row, int kernel_col, int kernel_units/*channels*/,
    int stride_row, int stride_col, int padding_row, int padding_col)
{
    assert(kernel_row > 0 && kernel_col > 0 && kernel_units > 0);
    assert(dst.rows() * dst.cols() > 1);
    int img_row, img_col;
    int adj_row, adj_col;

    //if (dst.empty())
    //{
        adj_row = (src_row + 2 * padding_row - kernel_row) % stride_row;
        adj_col = (src_col + 2 * padding_col - kernel_col) % stride_col;
        img_row = (src_row - 1) * stride_row + kernel_row - 2 * padding_row + adj_row;
        img_col = (src_col - 1) * stride_col + kernel_col - 2 * padding_col + adj_col;
        //dst = Matrix(img_row, img_col);
    //}

    //assert(dst.rows() == img_row && dst.cols() == img_col);
    col2im_gpu(src.GetDevMat(), kernel_units, input_row, input_col, kernel_row,
        stride_row, padding_row, dst.GetDevMat());
    dst.DataTransfer(DeviceToHost);
}

void Conv2D::Backward(Maxpooling* pre_layer, bool update, const int batch_size)
{
    if (!pre_layer)
    {
        printf("ERROR: pre_layer (Maxpooling*) in Conv2D::Backward is null pointer.\n");
        getchar();
        return;
    }

    int threads = 64;
    int blocks = (local_out.size() + threads - 1) / threads;//warning
    //calculate f'(x)
    gradient <<<blocks, threads >>> (local_out.GetDevMat(), differial.GetDevMat(), local_out.size());
    cudaDeviceSynchronize();

    //transpose colImg matrix
    mc.MatrixTranspose(colImg.GetDevMat(), colImg_t.GetDevMat(), colImg.rows(), colImg.cols());

    //transpose kernel
    mc.MatrixTranspose(kernel.GetDevMat(), kernel_t.GetDevMat(), kernel_row, kernel_col);

    //calculate loss matrix in previous layer, but this result cannot apply for calculating gradient in previous directly.
    //it needs reforming, so it can have the same rows and columns as output matrix in previous layer.
    if (loss_colImg.empty())
        loss_colImg = Matrix(kernel_t.rows(), loss.cols());
    mc.gemm_gpu(kernel_t.GetDevMat(), kernel_t.rows(), kernel_t.cols(), loss.GetDevMat(), loss.rows(), loss.cols(),
        loss_colImg.GetDevMat(), loss_colImg.rows(), CUBLAS_OP_N, CUBLAS_OP_N);

    //reform loss_colImg
    pre_layer->loss.Zeroreset();
    //col2im_gpu(loss_colImg.GetDevMat(), channels, loss_colImg.rows(), loss_colImg.cols(), kernel_row, stride_rows, padding_rows,
    //    pre_layer->loss.rows(), pre_layer->loss.cols(), pre_layer->loss.GetDevMat());
    col2im_gpu(loss_colImg.GetDevMat(), channels, input_row, input_col, kernel_row, stride_rows, padding_rows, pre_layer->loss.GetDevMat());
    pre_layer->loss.DataTransfer(DeviceToHost);

    mc.MatrixEleMult(loss.GetDevMat(), differial.GetDevMat(), loss.rows(), loss.cols());

    mc.gemm_gpu(loss.GetDevMat(), loss.rows(), loss.cols(), colImg_t.GetDevMat(), colImg_t.rows(), colImg_t.cols(), 
        grad_sample.GetDevMat(), grad_sample.rows(), CUBLAS_OP_N, CUBLAS_OP_N);
    kernel_grad += grad_sample;
    sample_num++;

    if (update)
    {
        assert(sample_num > 0);
        kernel -= kernel_grad * lr / (MYTYPE)sample_num;
        save_grad = kernel_grad / (MYTYPE)sample_num;
        kernel_grad.Zeroreset();
        kernel.DataTransfer(DeviceToHost);
        sample_num = 0;
    }
}

void Conv2D::Backward(Conv2D* pre_layer, bool update, const int batch_size)
{
    if (!pre_layer)
    {
        printf("ERROR: pre_layer (Conv2D*) in Conv2D::Backward is null pointer.\n");
        getchar();
        return;
    }
    
    int threads = 64;
    int blocks = (local_out.size() + threads - 1) / threads;
    //calculate f'(x)
    gradient <<<blocks,threads>>> (local_out.GetDevMat(), differial.GetDevMat(), local_out.size());
    cudaDeviceSynchronize();

    if (colImg_t.empty())
        colImg_t = Matrix(colImg.cols(), colImg.rows());
    //transpose colImg matrix
    mc.MatrixTranspose(colImg.GetDevMat(), colImg_t.GetDevMat(), colImg.rows(), colImg.cols());
    //colImg_t.DataTransfer(DeviceToHost);

    //transpose kernel
    mc.MatrixTranspose(kernel.GetDevMat(), kernel_t.GetDevMat(), kernel.rows(), kernel.cols());

    //calculate loss matrix in previous layer, but this result cannot apply for calculating gradient in previous directly.
    //it needs reforming, so it can have the same rows and columns as output matrix in previous layer.
    if (loss_colImg.empty())
        loss_colImg = Matrix(kernel_t.rows(), loss.cols());
    mc.gemm_gpu(kernel_t.GetDevMat(), kernel_t.rows(), kernel_t.cols(), loss.GetDevMat(), loss.rows(), loss.cols(),
        loss_colImg.GetDevMat(), loss_colImg.rows(), CUBLAS_OP_N, CUBLAS_OP_N);

    //reform loss_colImg
    pre_layer->loss.Zeroreset();
    col2im_gpu(loss_colImg.GetDevMat(), input_units, input_row, input_col, kernel_row, stride_rows, padding_rows, pre_layer->loss.GetDevMat());
    pre_layer->loss.DataTransfer(DeviceToHost);

    mc.MatrixEleMult(loss.GetDevMat(), differial.GetDevMat(), loss.rows(), loss.cols());
    mc.gemm_gpu(loss.GetDevMat(), loss.rows(), loss.cols(), colImg_t.GetDevMat(), colImg_t.rows(), colImg_t.cols(),
        grad_sample.GetDevMat(), grad_sample.rows(), CUBLAS_OP_N, CUBLAS_OP_N);
    kernel_grad += grad_sample;
    sample_num++;

    if (update)
    {
        assert(sample_num > 0);
        kernel -= kernel_grad * lr / (MYTYPE)sample_num;
        save_grad = kernel_grad / (MYTYPE)sample_num;
        kernel_grad.Zeroreset();
        kernel.DataTransfer(DeviceToHost);
        sample_num = 0;
    }
}

void Conv2D::Backward(bool update, const int batch_size)
{
    int threads = 64;
    int blocks = (local_out.size() + threads - 1) / threads;

    gradient <<<blocks, threads>>> (local_out.GetDevMat(), differial.GetDevMat(), local_out.size());
    cudaDeviceSynchronize();

    if (colImg_t.empty())
        colImg_t = Matrix(colImg.cols(), colImg.rows());
    mc.MatrixTranspose(colImg.GetDevMat(), colImg_t.GetDevMat(), colImg.rows(), colImg.cols());

    mc.MatrixEleMult(loss.GetDevMat(), differial.GetDevMat(), loss.rows(), loss.cols());
    mc.gemm_gpu(loss.GetDevMat(), loss.rows(), loss.cols(), colImg_t.GetDevMat(), colImg_t.rows(), colImg_t.cols(),
        grad_sample.GetDevMat(), grad_sample.rows(), CUBLAS_OP_N, CUBLAS_OP_N);
    kernel_grad += grad_sample;
    sample_num++;
    
    if (update)
    {
        assert(batch_size > 0);
        kernel -= kernel_grad * lr / (MYTYPE)sample_num;
        save_grad = kernel_grad / (MYTYPE)sample_num;
        kernel_grad.Zeroreset();
        sample_num = 0;
    }
}

void Conv2D::lrDecay(const int now_epoch)
{
    lr = lr * pow(0.96, (double)now_epoch / 5);
}

void Conv2D::Save(string& _dir, int which)
{
    FILE* fp = fopen((_dir + "/" + "Conv2D" + std::to_string(which) + ".txt").c_str(), "w");
    if (!fp)
    {
        getchar();
        return;
    }
    fprintf(fp, "lr=%f\n", lr);
    fprintf(fp, "kernel:\n");
    for (int i = 0; i < kernel.rows(); i++)
    {
        fprintf(fp, "row %d:\n", i);
        for (int j = 0; j < kernel.cols(); j++)
        {
            fprintf(fp, "%f ", kernel(i, j));
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\nloss matrix:\n");
    loss.DataTransfer(DeviceToHost);
    for (int i = 0; i < loss.rows(); i++)
    {
        fprintf(fp, "row %d:\n", i);
        for (int j = 0; j < loss.cols(); j++)
            fprintf(fp, "%f ", loss(i, j));
        fprintf(fp, "\n");
    }
    fprintf(fp, "\ngrad:\n");
    save_grad.DataTransfer(DeviceToHost);
    for (int i = 0; i < save_grad.rows(); i++)
    {
        fprintf(fp, "row %d:\n", i);
        for (int j = 0; j < save_grad.cols(); j++)
            fprintf(fp, "%f ", save_grad(i, j));
        fprintf(fp, "\n");
    }
    fclose(fp);
}

