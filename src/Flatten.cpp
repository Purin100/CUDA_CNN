#include "Flatten.h"

Flatten::Flatten()
{
    //output = nullptr;
    size = 0;
}

Flatten::~Flatten()
{
    // if(output)
    // {
    //     delete[] output;
    //     output = nullptr;
    // }
    /*if (loss)
    {
        delete[] loss;
        loss = nullptr;
    }*/
}

bool Flatten::BuildLayer(const int input_row, const int input_col)
{
    size = input_row * input_col;
    assert(size > 0);
    output = Vector(size);
    loss = Vector(size);
    reverse_output = Vector(size);
    return true;
}

void Flatten::Forward(Conv2D* _input, int num)
{
    if (!_input)
    {
        printf("ERROR: input argument (Conv2D*) of Flatten layer is null pointer.\n");
        getchar();
        return;
    }

    if (output.empty())
        output = Vector(_input->GetOutputRow() * _input->GetOutputCol());

    int count = 0;
    for(int i = 0; i < _input->GetOutputRow(); i++)
        for (int j = 0; j < _input->GetOutputCol(); j++)
        {
            output[count] = _input->GetOutput()(i, j);
            count++;
        }
    output.DataTransfer(HostToDevice);
    //output.showVector();
}

void Flatten::Forward(MYTYPE** _input, int row, int col)
{
    int count = 0;
    if(!_input)
    {
        printf("ERROR: _input is null pointer.\n");
        getchar();
        std::abort();
        return;
    }
    if(row <= 0 || col <= 0)
    {
        printf("ERROR: invalid paraments. Please check values in paraments row and col, ensure row > 0 and col > 0.\n");
        getchar();
        std::abort();
        return;
    }
    size = row * col;
    /*if (output.empty())
    {
        output = Vector(size);
    }*/
    if (output.size() != size)
    {
        output.Realloc(size);
    }

   /* if (loss.empty())
        loss = Vector(size);*/
    //if (loss.size() != size)
        //loss.Realloc(size);

    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
        {
            output[count] = _input[i][j];
            count++;
        }
    //for (int i = 0; i < row; i++)
    //{
    //    memcpy(&output[count], _input[i], sizeof(MYTYPE) * col);
    //    count += col;
    //}
    output.DataTransfer(HostToDevice);
}

void Flatten::Forward(Vector& _input)
{
    if (_input.empty())
    {
        printf("ERROR: input vector _input int Flatten::Forward(Vector&, const int) is empty.\n");
        getchar();
        return;
    }
    this->output = _input;
}

void Flatten::Forward(Maxpooling* _input, int num)
{
    int count = 0;
    if(!_input)
    {
        printf("ERROR: input argument (Maxpooling*) of Flatten layer is null pointer.\n");
        getchar();
        return;
    }
    if(num <= 0)
    {
        printf("ERROR: input argument num of Flatten layer should larger than zero.\n");
        getchar();
        return;
    }
    if(_input->OutRow() <= 0 || _input->OutCol() <= 0)
    {
        printf("ERROR: input argument (Maxpooling*) of Flatten layer is empty.\n");
        getchar();
        return;
    }

   /* if (output.empty())
        output = Vector(_input->OutRow() * _input->OutCol());*/
    for (int i = 0; i < _input->GetOutputRow(); i++)
        for (int j = 0; j < _input->GetOutputCol(); j++)
        {
            output[count] = _input->Getoutput()(i, j);
            count++;
        }
    output.DataTransfer(HostToDevice);

    //if (loss.empty())
    //    loss = Vector(_input->OutRow() * _input->OutCol());
    //if (loss.size() != _input->OutRow() * _input->OutCol())
    //    loss.Realloc(_input->OutRow() * _input->OutCol());
}

void Flatten::Backward(Vector& _loss)
{
    /*if (!_loss)
    {
        printf("ERROR: null pointer _loss in Flattem::Backward.\n");
        getchar();
        return;
    }

    for (int i = 0; i < size; i++)
        loss[i] = _loss[i];*/
    if (_loss.empty())
    {
        printf("ERROR: vector _loss in Flatten::Backward is empty.\n");
        getchar();
        return;
    }
}

void Flatten::Backward(Conv2D* pre_layer)
{
    if (!pre_layer)
    {
        printf("ERROR: invalid parament pre_layer (Conv2D*) in Flatten::Backward. The pointer is null pointer.\n");
        getchar();
        return;
    }

    loss.DataTransfer(DeviceToHost);
    //loss.showVector();
    memcpy(pre_layer->loss.GetMat(), loss.GetVec(), sizeof(MYTYPE) * loss.size());
    cudaMemcpy(pre_layer->loss.GetDevMat(), loss.GetDevVec(), sizeof(MYTYPE) * loss.size(), cudaMemcpyDeviceToDevice);
}

void Flatten::Backward(pMaxp pre_layer)
{
    if (!pre_layer)
    {
        printf("ERROR: invalid parament pre_layer (Maxpooling*) in Flatten::Backward. The pointer is null pointer.\n");
        getchar();
        return;
    }
    loss.DataTransfer(DeviceToHost);
    //loss.showVector();
    memcpy(pre_layer->loss.GetMat(), loss.GetVec(), sizeof(MYTYPE) * loss.size());
    cudaMemcpy(pre_layer->loss.GetDevMat(), loss.GetDevVec(), sizeof(MYTYPE) * loss.size(), cudaMemcpyDeviceToDevice);
    //pre_layer->loss.showMat();
}

void Flatten::Reverse(Vector reverse_vec)
{
    reverse_vec.DataTransfer(DeviceToHost);
    reverse_output = reverse_vec;
}

void Flatten::DisplayOutput()
{
    //int zeros=0;
    for(int i = 0; i < size; i++)
    {
        printf("%f ", output[i]);
        //if(output[i]==0.0f)
        //    zeros++;
    }
    //printf("\nzeros:%d\n",zeros);
}

void Flatten::Save(const char* dir_name, const char* mode)
{
    FILE* fp;
    fp = fopen(dir_name, mode);
    if(!fp)
        return;
    for(int i = 0; i < size; i++)
    {
        fprintf(fp, "%f\n", output[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
}