#include "Net.h"

Net::Net(const int samples, const int batch_size)
{
    assert(samples > 0);
    this->batch_size = batch_size > 0 ? batch_size : 1;
    if (samples % batch_size == 0)
        batches = samples / batch_size;
    else
    {
        batches = samples / batch_size + 1;
        rest_sample = samples % batch_size;
    }
}

Net::~Net()
{

}

void Net::Forward(MYTYPE* _input, const int size)
{
    Dense* dlayer_now = nullptr, * dlayer_pre = nullptr;
    Conv2D* clayer_now = nullptr, *clayer_pre = nullptr;
    //Maxpooling* mlayer_now;
    //AveragePooling* alayer_now = nullptr;//not in use for now
    Flatten* flayer_now = nullptr;

    if (!_input)
    {
        printf("ERROR: empty pointer _image.\n");
        getchar();
        abort();
    }

    if (input.empty())
        input = Vector(_input, size);

    auto now_layer = layers.begin();
    auto pre_layer = layers.begin();

    if (now_layer->type == DENSE)
        reinterpret_cast<Dense*>(now_layer->layer)->Forward(input, size);
    now_layer++;

    int preType;
    while (now_layer < layers.end())
    {
        preType = pre_layer->type;

        switch (now_layer->type)
        {
        case DENSE:
            dlayer_now = (Dense*)((*now_layer).layer);
            if (preType == DENSE)
            {
                dlayer_pre = reinterpret_cast<Dense*>(pre_layer->layer);
                dlayer_now->Forward(dlayer_pre);
            }
            break;
        case FLATTEN:
            flayer_now = (Flatten*)((*now_layer).layer);
            if (preType == CONV2D)
            {
                //flayer_now->Forward()
            }
            break;

        }

        now_layer++;
        pre_layer++;
    }
}

void Net::Forward(TXTReader* _input)
{
    if (!_input)
    {
        printf("ERROR: parament _input is null pointer.\n");
        getchar();
        return;
    }
    Conv2D* clayer_now = nullptr, * clayer_pre = nullptr;
    Dense* dlayer_now = nullptr, * dlayer_pre = nullptr;
    Flatten* flayer_now = nullptr;
    pMaxp player_now = nullptr;
    Dropout* drlayer_now = nullptr;
    auto now_layer = layers.begin();
    auto pre_layer = layers.begin();

    if (now_layer->type == FLATTEN)
        reinterpret_cast<Flatten*>(now_layer->layer)->Forward(_input->Getdata(0), _input->Width(), _input->Height());
    if (now_layer->type == CONV2D)
        reinterpret_cast<Conv2D*>(now_layer->layer)->Forward(_input->Getdata(0), _input->Width(), _input->Height());
    now_layer++;

    int preType;
    while (now_layer < layers.end())
    {
        preType = pre_layer->type;

        switch (now_layer->type)
        {
        case DENSE:
            dlayer_now = reinterpret_cast<Dense*>((*now_layer).layer);
            if (preType == DENSE)
            {
                dlayer_pre = reinterpret_cast<Dense*>(pre_layer->layer);
                dlayer_now->Forward(dlayer_pre);
            }
            if (preType == FLATTEN)
            {
                dlayer_now->Forward(reinterpret_cast<Flatten*>(pre_layer->layer)->GetOutput(),
                    reinterpret_cast<Flatten*>(pre_layer->layer)->GetSize());
            }
            if (preType == DROPOUT)
            {
                dlayer_now->Forward(reinterpret_cast<Dropout*>(pre_layer->layer)->GetOutput(),
                    reinterpret_cast<Dropout*>(pre_layer->layer)->Getsize());
            }
            break;
        case FLATTEN:
            flayer_now = (Flatten*)((*now_layer).layer);
            if (preType == CONV2D)
            {
                flayer_now->Forward(reinterpret_cast<Conv2D*>(pre_layer->layer));
            }
            if (preType == MAXPOOLING)
            {
                flayer_now->Forward(reinterpret_cast<pMaxp>(pre_layer->layer));
            }
            break;
        case CONV2D:
            clayer_now = (Conv2D*)(now_layer->layer);
            if (preType == CONV2D)
            {
                clayer_now->Forward(reinterpret_cast<Conv2D*>(pre_layer->layer));
            }
            if (preType == MAXPOOLING)
            {
                clayer_now->Forward(reinterpret_cast<pMaxp>(pre_layer->layer));
            }
            break;
        case MAXPOOLING:
            player_now = reinterpret_cast<pMaxp>(now_layer->layer);
            
            if (preType == CONV2D)
            {
                clayer_pre = reinterpret_cast<Conv2D*>(pre_layer->layer);
                Matrix in;
                if (player_now->unit() == 1)
                {
                    in = Matrix(clayer_pre->GetOutput().rows(), clayer_pre->GetOutput().cols());
                    //the output matrix from Conv2D is column-major, which is different from the input matrix required by Maxpooling layer (row-major matrix)
                    //so we need to convert output matrix to row-major matrix
                    Colmaj2Rowmaj(clayer_pre->GetOutput().GetMat(), in.GetMat(), clayer_pre->GetOutput().rows(),
                        clayer_pre->GetOutput().cols()); 
                }
                else
                {
                    in = Matrix(clayer_pre->GetOutputRow(), clayer_pre->GetOutputCol());

                    //the output matrix from Conv2D is column-major, which is different from the input matrix required by Maxpooling layer (row-major matrix)
                    //so we need to convert output matrix to row-major matrix
                    Colmaj2Rowmaj(clayer_pre->GetOutput().GetMat(), in.GetMat(), clayer_pre->GetOutputRow(),
                        clayer_pre->GetOutputCol());
                }

                player_now->Forward(in);
            }
            break;
        case DROPOUT:
            drlayer_now = reinterpret_cast<Dropout*>(now_layer->layer);
            if (preType == DENSE)
            {
                dlayer_pre = reinterpret_cast<Dense*>(pre_layer->layer);
                drlayer_now->Forward(dlayer_pre->Getoutput(), train);
            }
            if (preType == FLATTEN)
            {
                flayer_now = reinterpret_cast<Flatten*>(pre_layer->layer);
                drlayer_now->Forward(flayer_now->GetOutput(), train);
            }
            break;
        }

        now_layer++;
        pre_layer++;
    }
}

void Net::Forward(Vector& _input, int row, int col, int channel)
{
    Conv2D* clayer_now = nullptr, * clayer_pre = nullptr;
    Dense* dlayer_now = nullptr, * dlayer_pre = nullptr;
    Flatten* flayer_now = nullptr;
    pMaxp player_now = nullptr;
    auto now_layer = layers.begin();
    auto pre_layer = layers.begin();

    if (now_layer->type == CONV2D)
        reinterpret_cast<Conv2D*>(now_layer->layer)->Forward(_input, row, col, channel);
    if (now_layer->type == FLATTEN)
        reinterpret_cast<Flatten*>(now_layer->layer)->Forward(_input);
    if (now_layer->type == DENSE)
    {
        this->input = _input;
        reinterpret_cast<Dense*>(now_layer->layer)->Forward(input, row * col);
    }
    now_layer++;

    LayerType preType;
    while (now_layer < layers.end())
    {
        preType = pre_layer->type;

        switch (now_layer->type)
        {
        case DENSE:
            dlayer_now = (Dense*)((*now_layer).layer);
            if (preType == DENSE)
            {
                dlayer_pre = reinterpret_cast<Dense*>(pre_layer->layer);
                dlayer_now->Forward(dlayer_pre);
            }
            if (preType == FLATTEN)
            {
                dlayer_now->Forward(reinterpret_cast<Flatten*>(pre_layer->layer)->GetOutput(),
                    reinterpret_cast<Flatten*>(pre_layer->layer)->GetSize());
            }
            break;
        case FLATTEN:
            flayer_now = (Flatten*)((*now_layer).layer);
            if (preType == CONV2D)
            {
                flayer_now->Forward(reinterpret_cast<Conv2D*>(pre_layer->layer));
            }
            if (preType == MAXPOOLING)
            {
                flayer_now->Forward(reinterpret_cast<pMaxp>(pre_layer->layer));
            }
            break;
        case CONV2D:
            clayer_now = (Conv2D*)(now_layer->layer);
            if (preType == CONV2D)
            {
                clayer_now->Forward(reinterpret_cast<Conv2D*>(pre_layer->layer));
            }
            if (preType == MAXPOOLING)
            {
                clayer_now->Forward(reinterpret_cast<pMaxp>(pre_layer->layer));
            }
            break;
        case MAXPOOLING:
            player_now = (pMaxp)(now_layer->layer);
            if (preType == CONV2D)
                player_now->Forward(reinterpret_cast<Conv2D*>(pre_layer->layer)->GetOutput());
            break;
        }

        now_layer++;
        pre_layer++;
    }
}

void Net::Backward(Vector onehot_label)
{
    if (onehot_label.empty())
    {
        printf("ERROR: parament onehot_label is empty.\n");
        getchar();
        return;
    }

    int category = layers.back().num;
    Vector last_out = GetLayerOutput(layers.back().order);

    loss = Cross_Entropy(onehot_label.GetVec(), last_out.GetVec(), category);
    total_loss += loss;

    //gradient for Softmax function
    last_out -= onehot_label;

    int batch_sample_num = 0;
    this->current_sample++;
    this->update = false;
    if (this->current_sample == batch_size)
    {
        this->update = true;
        batch_sample_num = current_sample;
        current_batch++;
        this->current_sample = 0;
    }
    if (rest_sample && this->current_batch == batches && current_sample == rest_sample-1)
    {
        if (current_sample == rest_sample)
        {
            this->update = true;
            batch_sample_num = current_sample;
            current_batch = 1;
            this->current_sample = 0;
        }
    }


    //BP starts here
    auto now_layer = layers.end() - 1;
    auto pre_layer = layers.end() - 2;

    while (now_layer >= layers.begin())
    {
        //The last layer has different inputs, so it will be a seperate branch
        if (now_layer == layers.end() - 1)
        {
            if (now_layer->type == DENSE && pre_layer->type == DENSE)
            {
                {
                    reinterpret_cast<Dense*>(now_layer->layer)->Backward(last_out, reinterpret_cast<Dense*>(pre_layer->layer), this->update, batch_sample_num);
                }
            }
            if (now_layer->type == DENSE && pre_layer->type == DROPOUT)
            {
                reinterpret_cast<Dense*>(now_layer->layer)->loss = last_out;
                reinterpret_cast<Dense*>(now_layer->layer)->Backward(reinterpret_cast<Dropout*>(pre_layer->layer));
            }
            now_layer--;
            if (pre_layer != layers.begin())
                pre_layer--;
            continue;
        }

        if (now_layer == layers.begin())
        {
            if (now_layer->type == CONV2D)
                reinterpret_cast<Conv2D*>(now_layer->layer)->Backward(this->update, batch_sample_num);
            break;
        }

        //Other layers' BP
        switch (now_layer->type)
        {
        case DENSE:
            if (now_layer == layers.begin())
            {
                reinterpret_cast<Dense*>(now_layer->layer)->Backward(input, this->update, batch_sample_num);
                break;
            }
            if (pre_layer->type == DENSE)
            {
                reinterpret_cast<Dense*>(now_layer->layer)->Backward(reinterpret_cast<Dense*>(pre_layer->layer), this->update, batch_sample_num);
            }
            if (pre_layer->type == FLATTEN)
            {
                reinterpret_cast<Dense*>(now_layer->layer)->Backward(reinterpret_cast<Flatten*>(pre_layer->layer), this->update, batch_sample_num);
            }
            if (pre_layer->type == DROPOUT)
                reinterpret_cast<Dense*>(now_layer->layer)->Backward(reinterpret_cast<Dropout*>(pre_layer->layer));
            break;
        case FLATTEN:
            if (pre_layer->type == MAXPOOLING)
                reinterpret_cast<Flatten*>(now_layer->layer)->Backward(reinterpret_cast<pMaxp>(pre_layer->layer));
            if (pre_layer->type == CONV2D)
                reinterpret_cast<Flatten*>(now_layer->layer)->Backward(reinterpret_cast<Conv2D*>(pre_layer->layer));
            break;
        case MAXPOOLING:
            if (pre_layer->type == CONV2D)
                reinterpret_cast<pMaxp>(now_layer->layer)->Backward(reinterpret_cast<Conv2D*>(pre_layer->layer)->loss);
            break;
        case CONV2D:
            if (pre_layer->type == CONV2D)
                reinterpret_cast<Conv2D*>(now_layer->layer)->Backward(reinterpret_cast<Conv2D*>(pre_layer->layer), update, batch_sample_num);
            if (pre_layer->type == MAXPOOLING)
                reinterpret_cast<Conv2D*>(now_layer->layer)->Backward(reinterpret_cast<pMaxp>(pre_layer->layer), update, batch_sample_num);
            break;
        case DROPOUT:
            if (pre_layer->type == DENSE)
                reinterpret_cast<Dropout*>(now_layer->layer)->Backward(reinterpret_cast<Dense*>(pre_layer->layer)->loss);
            if (pre_layer->type == FLATTEN)
                reinterpret_cast<Flatten*>(pre_layer->layer)->loss = reinterpret_cast<Dropout*>(now_layer->layer)->loss;
            break;
        default:
            break;
        }

        //if (now_layer == layers.begin())
            //break;
        now_layer--;
        if (pre_layer != layers.begin())
            pre_layer--;
        //else//If pre_layer == layers.begin(), BP ends.
            //break;
    }
}

bool Net::Eval(int label, Vector onehot_label)
{
    MYTYPE max = -99.0;
    int predict = -1;
    int category = layers.back().num;
    Vector last_out = GetLayerOutput(layers.back().order);

    //calculate loss value
    loss = Cross_Entropy(onehot_label.GetVec(), last_out.GetVec(), category);
    total_loss += loss;


    for (int i = 0; i < category; i++)
    {
        if (last_out[i] > max)
        {
            max = last_out[i];
            predict = i;
        }
    }
    if (predict >= 0 && predict < category)
        cate_res[predict]++;
    else
    {
        printf("ERROR: invalid predict result!\n");
        getchar();
        exit(2);
    }
    confuse[label][predict]++;
    return (predict == label);
}

void Net::Save(std::string _dir)
{
//create floders for saving
#if defined(_WIN32) || defined(_WIN64)
    if (_access(_dir.c_str(), 0) != 0)
        if (_mkdir(_dir.c_str()) != 0)
        {
            printf("Create directory failed.\n");
            return;
        }
#endif
#ifdef linux
    if (access(_dir.c_str(), 0) != 0)
        if (mkdir(_dir.c_str(), 0777) < 0)
        {
            printf("Create directory failed.\n");
            return;
        }
#endif

    auto layer = layers.begin();
    while (layer < layers.end())
    {
        switch (layer->type)
        {
        case DENSE:
            reinterpret_cast<Dense*>(layer->layer)->Save(_dir, layer->order);
            break;
        case CONV2D:
            reinterpret_cast<Conv2D*>(layer->layer)->Save(_dir, layer->order);
            break;
        case MAXPOOLING:
            reinterpret_cast<pMaxp>(layer->layer)->Save(_dir, layer->order);
            break;
        default:
            break;
        }
        layer++;
    }

    FILE* fp = fopen((_dir + "/confuse matrix.txt").c_str(), "w");
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
            fprintf(fp, "%d ", confuse[i][j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

const Vector Net::GetLayerOutput(int which)
{
    
    if (which<0 || which>order)
    {
        printf("ERROR: parament which is not in the range of [0, %d).\n", order);
        getchar();
        abort();
    }
    if (layers[which].type == DENSE)
        return reinterpret_cast<Dense*>(layers[which].layer)->Getoutput();
    
}

void Net::lrDecay(const int now_epoch)
{
    if (layers.empty())
        return;
    auto now_layer = layers.begin();

    while (now_layer < layers.end())
    {
        if (now_layer->type == DENSE)
            reinterpret_cast<Dense*>(now_layer->layer)->lrDecay(now_epoch);
        if (now_layer->type == CONV2D)
            reinterpret_cast<Conv2D*>(now_layer->layer)->lrDecay(now_epoch);
        now_layer++;
    }
}
