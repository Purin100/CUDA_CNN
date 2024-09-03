#include "Dense.cuh"
#include <ctime>

//static std::uniform_real_distribution<MYTYPE> lr_dis(0.0001, 0.001);
static std::bernoulli_distribution ber(1.0);
static std::random_device dense_rd;
//static std::mt19937_64 dense_weight(dense_rd());
static std::uniform_real_distribution<MYTYPE> xavier(-1.0,1.0);

Dense::~Dense()
{
    if (adam)
    {
        delete adam;
        adam = nullptr;
    }
}

bool Dense::BuildLayer(int input_units, int output_units, const char* activation, const char* optimizer,
    MYTYPE drop_rate, bool isfreeze)
{
    if (activation == nullptr)
    {
        printf("ERROR: null pointer activation.\n");
        getchar();
        return false;
    }
    if (input_units < 0 || output_units <= 0)
    {
        printf("ERROR: invalid arguments. Please ensure input_units >= 0, and output_units > 0.");
        getchar();
        return false;
    }
    if (drop_rate > 0.0 && drop_rate < 1.0)
    {
        this->drop_rate = drop_rate;
        //drop = std::uniform_real_distribution<MYTYPE>(0.0, drop_rate);
    }
    else
        this->drop_rate = 0.0;

    this->input_units = input_units;
    this->output_units = output_units;
    this->freeze_weight = isfreeze;

    weight = Matrix(output_units, input_units);
    weight_grad = Matrix(output_units, input_units);
    grad_sample = Matrix(output_units, input_units);
    weight_t = Matrix(input_units, output_units);
    save_grad = Matrix(output_units, input_units);

    output = Vector(output_units);
    loss = Vector(output_units);
    bias = Vector(output_units);
    bias_batch = Vector(output_units);
        
    local_out = Vector(output_units);

    input = Vector(input_units);


    //input_frequence = Vector(input_units);
    //output_frequence = Vector(output_units);

    //Set activation function and relevant gradient function
    //Softmax DO NOT have gradient function, and cannot be used in hidden layers
    if (mystrcmp(activation, "tanh") == 0)
    {
        this->activation = &Tanh;
        this->gradient = &Tanh_gradient;
        goto ENDACTIVATON;
    }
    if (mystrcmp(activation, "relu") == 0)
    {
        this->activation = &Relu;
        this->gradient = &Relu_gradient;
        goto ENDACTIVATON;
    }
    if (mystrcmp(activation, "relu1") == 0)
    {
        this->activation = &Relu1;
        this->gradient = &Relu1_gradient;
        goto ENDACTIVATON;
    }
    if (mystrcmp(activation, "softmax") == 0)
    {
        this->activation = &Softmax;
        goto ENDACTIVATON;
    }
    if (mystrcmp(activation, "sigmoid") == 0)
    {
        this->activation = &Sigmoid;
        this->gradient = &Sigmoid_gradient;
        goto ENDACTIVATON;
    }
    if (mystrcmp(activation, "leakyrelu") == 0)
    {
        this->activation = &LeakyRelu;
        this->gradient = &LeakyRelu_gradient;
        goto ENDACTIVATON;
    }
    if (mystrcmp(activation, "linear") == 0)
    {
        this->activation = &Linear;
        this->gradient = &Linear_gradient;
        goto ENDACTIVATON;
    }
ENDACTIVATON:
    //Initialize weight matrix and bias vector
    MYTYPE t =  sqrt(6.0 / (input_units + output_units));
    
    for (int i = 0; i < weight.rows(); i++)
        for (int j = 0; j < weight.cols(); j++)
        {
            weight(i, j) = xavier(dense_rd) * t;
        }
    weight.DataTransfer(HostToDevice);
    for (int i = 0; i < bias.size(); i++)
        bias[i] = 0.0;
    bias.DataTransfer(HostToDevice);
    if ((optimizer) && mystrcmp(optimizer, "adam") == 0)
    {
        adam = new Adam;
        if (!adam->Init(output_units, input_units, "matrix", lr))
            return false;
    }
    weight_grad.Zeroreset();
    bias_batch.ZeroReset();
    save_grad.Zeroreset();
    return true;
}

void Dense::Forward(Dense* pre_layer)
{
    //cudaError_t status;
    if (!pre_layer)
    {
        printf("ERROR: pre_layer is null pointer.\n");
        getchar();
        return;
    }
;
    int threads = 32;
    int blocks;


    //z = Wx + b
    mc.gemv_gpu(weight.GetDevMat(), weight.rows(), weight.cols(), pre_layer->output.GetDevVec(), local_out.GetDevVec());
    local_out += bias;
   
    blocks = (output_units + threads - 1) / threads;
    if (activation != &Softmax)
    {
        //a = activation(z)
        activation <<<blocks, threads >>> (local_out.GetDevVec(), output.GetDevVec(), output_units);
        cudaDeviceSynchronize();
        output.DataTransfer(DeviceToHost);
    }
    else//Softmax works on CPU
    {
        local_out.DataTransfer(DeviceToHost);
        activation(local_out.GetVec(), output.GetVec(), output_units);
        output.DataTransfer(HostToDevice);
    }
}

void Dense::Forward(Vector& _input, const int element_num)
{
    if (_input.empty())
    {
        printf("ERROR: input vector _input is empty.\n");
        getchar();
        return;
    }
    
    int threads = 32;
    int blocks;

    blocks = (output.size() + threads - 1) / threads;
    //z = Wx + b
    mc.gemv_gpu(weight.GetDevMat(), output_units, input_units, _input.GetDevVec(), local_out.GetDevVec());
    local_out += bias;

    if (activation != Softmax)
    {
        activation << <blocks, threads >> > (local_out.GetDevVec(), output.GetDevVec(), output_units);
        cudaDeviceSynchronize();
        output.DataTransfer(DeviceToHost);
    }
    else//Softmax works on CPU
    {
        local_out.DataTransfer(DeviceToHost);
        activation(local_out.GetVec(), output.GetVec(), output_units);
        output.DataTransfer(HostToDevice);
    }   
}

void Dense::Backward(Vector& _loss, Dense* pre_layer, bool update, const int batch_size)
{
    if (_loss.empty())
    {
        printf("ERROR: invalide parament(s). Please double check whether batch_size > 0, _loss is not empty.\n");
        getchar();
        return;
    }
    if (!pre_layer)
    {
        printf("ERROR: parament pre_layer is null pointer.\n");
        getchar();
        return;
    }

    //Transposition weight matrix
    mc.MatrixTranspose(weight.GetDevMat(), weight_t.GetDevMat(), output_units, input_units);

    // In BP, we will update weight matrix and bias vector
    if(!freeze_weight)
        bias_batch += _loss;
    //loss in previous layer delta' = transposition(W) * loss
    mc.gemv_gpu(weight_t.GetDevMat(), input_units, output_units, _loss.GetDevVec(), pre_layer->loss.GetDevVec());

    if (freeze_weight)
        return;
    //For the last layer, the gradient of Softmax + Cross Entropy already calculated in Net::Backward()
    //so calculate delta = loss * x directly
    mc.gemm_gpu(_loss.GetDevVec(), output_units, 1, pre_layer->output.GetDevVec(), 1, input_units,
        grad_sample.GetDevMat(), output_units, CUBLAS_OP_N, CUBLAS_OP_N);
    weight_grad += grad_sample;
    sample_num++;

    if (update)
    {
        if (adam && adam->InitState())
            adam->Update(weight_grad / (MYTYPE)sample_num, weight);
        else
            weight -= weight_grad * lr / (MYTYPE)sample_num;
        bias -= bias_batch * lr / (MYTYPE)sample_num;
        save_grad = weight_grad / (MYTYPE)sample_num;
        weight_grad.Zeroreset();
        bias_batch.ZeroReset();
        sample_num = 0;
    }
    this->loss = _loss;
}

void Dense::Backward(Dense* pre_layer, bool update, const int batch_size)
{
    Vector differal(output_units);
    if (!pre_layer)
    {
        printf("ERROR: parament pre_layer is null pointer.\n");
        getchar();
        return;
    }

    mc.MatrixTranspose(weight.GetDevMat(), weight_t.GetDevMat(), output_units, input_units);
    //weight_t.DataTransfer(DeviceToHost);

    //gradient = loss * f'(a) * x
    int threads = 32;
    int blocks;

    blocks = (output_units + threads - 1) / threads;
    //calculate f'(a)
    //gradient<<<blocks,threads>>>(output.GetDevVec(), differal.GetDevVec(), output_units);
    gradient <<<blocks, threads >>> (local_out.GetDevVec(), differal.GetDevVec(), output_units);
    cudaDeviceSynchronize();

    if(!freeze_weight)
        bias_batch += loss;
    //loss * f'(a) 
    // loss value here should multiply with differal before update weights in this layer
    mc.VecEleMult(loss.GetDevVec(), differal.GetDevVec(), output_units);

    //calculate loss value in previous layer
    mc.gemv_gpu(weight_t.GetDevMat(), input_units, output_units, loss.GetDevVec(), pre_layer->loss.GetDevVec());

    if (freeze_weight)
        return;
    //calculate graident
    /*mc.gemm_gpu(loss.GetDevVec(), output_units, 1, pre_layer->output.GetDevVec(), 1, pre_layer->output_units,
        weight_grad.GetDevMat(), weight_grad.rows(), CUBLAS_OP_N, CUBLAS_OP_N);*/
    mc.gemm_gpu(loss.GetDevVec(), output_units, 1, pre_layer->output.GetDevVec(), 1, input_units,
        grad_sample.GetDevMat(), output_units, CUBLAS_OP_N, CUBLAS_OP_N);
    weight_grad += grad_sample;
    sample_num++;

    if (update)
    {
        if (adam && adam->InitState())
            adam->Update(weight_grad / (MYTYPE)sample_num, weight);
        else
            weight -= weight_grad * lr / (MYTYPE)sample_num;
        bias -= bias_batch * lr / (MYTYPE)sample_num;
        save_grad = weight_grad / (MYTYPE)sample_num;
        weight_grad.Zeroreset();
        bias_batch.ZeroReset();
        sample_num = 0;
    }
}

void Dense::Backward(Flatten* pre_layer, bool update, const int batch_size)
{
    Vector differal(output_units);

    if (!pre_layer)
    {
        printf("ERROR: pre_layer (Flatten*) in Dense::Backward is null pointer.\n");
        getchar();
        return;
    }

    mc.MatrixTranspose(weight.GetDevMat(), weight_t.GetDevMat(), output_units, input_units);
    //weight_t.DataTransfer(DeviceToHost);

    int threads = 32;
    int blocks;

    blocks = (output_units + threads - 1) / threads;
    //calculate f'(a)
    /*gradient <<<blocks, threads >>> (output.GetDevVec(), differal.GetDevVec(), output_units);*/
    gradient <<<blocks, threads >>> (local_out.GetDevVec(), differal.GetDevVec(), output_units);
    cudaDeviceSynchronize();

    if(!freeze_weight)
        bias_batch += loss;
    //loss * f'(a)
    mc.VecEleMult(loss.GetDevVec(), differal.GetDevVec(), output_units);

    //calculate loss value in previous layer
    mc.gemv_gpu(weight_t.GetDevMat(), input_units, output_units, loss.GetDevVec(), pre_layer->loss.GetDevVec());

    if (freeze_weight)
        return;
    //calculate graident
    //mc.gemm_gpu(loss.GetDevVec(), output_units, 1, pre_layer->GetOutput().GetDevVec(), 1, pre_layer->GetSize(),
    //    weight_grad.GetDevMat(), weight_grad.rows(), CUBLAS_OP_N, CUBLAS_OP_N);
    mc.gemm_gpu(loss.GetDevVec(), output_units, 1, pre_layer->output.GetDevVec(), 1, input_units,
        grad_sample.GetDevMat(), output_units, CUBLAS_OP_N, CUBLAS_OP_N);
    weight_grad += grad_sample;
    
    sample_num++;

    if (update)
    {
        if (adam && adam->InitState())
            adam->Update(weight_grad / (MYTYPE)batch_size, weight);
        else
            weight -= weight_grad * lr / (MYTYPE)batch_size;
        bias -= bias_batch * lr / (MYTYPE)batch_size;
        save_grad = weight_grad / (MYTYPE)batch_size;
        weight_grad.Zeroreset();
        bias_batch.ZeroReset();
        sample_num = 0;
    }

}

void Dense::Backward(Dropout* pre_layer)
{
    Vector differal(output_units);
    if (!pre_layer)
    {
        printf("ERROR: pre_layer in Dense::backward(Dropout*) is null pointer.\n");
        getchar();
        return;
    }
    mc.MatrixTranspose(weight.GetDevMat(), weight_t.GetDevMat(), output_units, input_units);

    int threads = (512);
    int blocks = ((output.size() + threads - 1) / threads);

    //calculate f'(a)
    gradient << <blocks, threads >> > (output.GetDevVec(), differal.GetDevVec(), output_units);
    cudaDeviceSynchronize();

    //loss * f'(a)
    mc.VecEleMult(loss.GetDevVec(), differal.GetDevVec(), output_units);

    //calculate loss value in previous layer
    mc.gemv_gpu(weight_t.GetDevMat(), weight_t.rows(), weight_t.cols(), loss.GetDevVec(), pre_layer->loss.GetDevVec());
    loss.DataTransfer(DeviceToHost);

    //calculate graident
    /*mc.gemm_gpu(loss.GetDevVec(), output_units, 1, pre_layer->GetOutput().GetDevVec(), 1, pre_layer->Getsize(),
        weight_grad.GetDevMat(), weight_grad.rows(),CUBLAS_OP_N,CUBLAS_OP_N);*/
    mc.gemm_gpu(loss.GetDevVec(), output_units, 1, pre_layer->GetOutput().GetDevVec(), 1, pre_layer->Getsize(),
        grad_sample.GetDevMat(), output_units, CUBLAS_OP_N, CUBLAS_OP_N);
    weight_grad += grad_sample;
    bias_batch += loss;
    sample_num++;

    weight -= weight_grad / sample_num * lr;
    //bias -= loss * lr;
    bias -= bias_batch / sample_num * lr;
    weight_grad.Zeroreset();
    bias_batch.ZeroReset();
    sample_num = 0;
}

void Dense::Backward(Vector& _input, bool update, const int batch_size)
{
    Vector differal(output_units);
    if (_input.empty())
    {
        printf("ERROR: empty vector _input in Dense::Backward(Vector&).\n");
        getchar();
        return;
    }

    mc.MatrixTranspose(weight.GetDevMat(), weight_t.GetDevMat(), output_units, input_units);

    int threads = 32;
    int blocks = (output.size() + threads - 1) / threads;

    //calculate f'(a)
    gradient <<<blocks, threads >>> (local_out.GetDevVec(), differal.GetDevVec(), output_units);
    cudaDeviceSynchronize();

    if (!freeze_weight)
        bias_batch += loss;
    //loss * f'(a)
    loss.DataTransfer(DeviceToHost);
    mc.VecEleMult(loss.GetDevVec(), differal.GetDevVec(), output_units);

    //calculate graident
    mc.gemm_gpu(loss.GetDevVec(), output_units, 1, _input.GetDevVec(), 1, input_units,
        grad_sample.GetDevMat(), output_units, CUBLAS_OP_N, CUBLAS_OP_N);
    //weight_grad.DataTransfer(DeviceToHost);
    weight_grad += grad_sample;

    sample_num++;
    if (update)
    {
        save_grad = weight_grad;
        if (adam->InitState())
            adam->Update(weight_grad / (MYTYPE)sample_num, weight);
        else
            weight -= weight_grad * lr / (MYTYPE)sample_num;
        bias -= bias_batch * lr / (MYTYPE)sample_num;

        weight_grad.Zeroreset();
        bias_batch.ZeroReset();
        sample_num = 0;
    }
    //weight.DataTransfer(DeviceToHost);
}

void Dense::Save(string& _dir, int which)
{
    FILE* fp = fopen((_dir + "/" + "Dense" + std::to_string(which) + ".txt").c_str(), "w");
    if (!fp)
    {
        printf("Cannot open file %s\n", (_dir + "/" + "Dense" + std::to_string(which) + ".txt").c_str());
        getchar();
        return;
    }


    fprintf(fp, "lr=%f\n", lr);

    for (int i = 0; i < weight.rows(); i++)
    {
        for (int j = 0; j < weight.cols(); j++)
            fprintf(fp, "%.7f ", weight(i, j));
        fprintf(fp, "\n");
    }

    fprintf(fp, "bias:\n");
    bias.DataTransfer(DeviceToHost);
    for (int i = 0; i < bias.size(); i++)
        fprintf(fp, "%f\n", bias[i]);
    
    fprintf(fp, "grad:\n");
    //weight_grad.DataTransfer(DeviceToHost);
    for (int i = 0; i < output_units; i++)
    {
        for (int j = 0; j < input_units; j++)
            fprintf(fp, "%.10f ", save_grad(i, j));
        fprintf(fp, "\n");
    }


    fprintf(fp, "loss:\n");
    loss.DataTransfer(DeviceToHost);
    for (int i = 0; i < loss.size(); i++)
    {
        fprintf(fp, "%.10f ", loss[i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void Dense::ExportWeight(Matrix* w)
{
    if (!w)
    {
        printf("ERROR: null pointer w (Matrix*) in Dense::ExportWeight.\n");
        getchar();
        return;
    }
    *w = this->weight;
}

void Dense::lrDecay(const int now_epoch)
{
#ifdef lr_num
    lr *= pow(0.98, now_epoch);
#endif
#ifdef lr_mat
    lr *= pow(0.98, now_epoch);
#endif
}