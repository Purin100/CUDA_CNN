/*
This file defines Net class which is the basic framework for the neural network in this project.
*/
#pragma once
#include "LayerInfo.h"
#include "utils.h"
#include "Dense.cuh"
#include "Flatten.h"
#include "TXTReader.h"
#include "Conv2D.cuh"
#include "Pooling.cuh"
#include "Dropout.h"
#include <vector>
#include <thread>

//The two #if are used for saving parameters in the network
#if defined(_WIN32) || defined(_WIN64)
#include <direct.h>
#include <io.h>
#endif/*Windows*/
#if defined linux
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif/*Linux*/

using std::vector;
using std::thread;


class Net
{
public:
    Net() {};
    Net(const int samples, const int batch_size);
    ~Net();

    void Forward(MYTYPE* _input, const int size);
    void Forward(TXTReader* _input);
    //void Forward(MYTYPE* _input, int row, int col);
    void Forward(Vector& _input, int row, int col, int channel = 1);

    void Backward(Vector onehot_label);
    bool Eval(int label, Vector onehot_label);

    void lrDecay(const int now_epoch);

    void Save(std::string _dir);
    inline void epochReset() { current_batch = 1; };

    //Get output in a sepcific layer
    //which >= 0 && which < this->order
    const Vector GetLayerOutput(int which);

    //From here, you will see three functions called add(),
    //different add function allows you adding layers in different ways.
    //But when adding the first layer to the neural network, you must use add(T*, L*), 
    //which requires you send a pointer of layer object and a pointer for revalant LayerInfo structure (defined in LayerInfo.h)
    //For example, you want to add a Dense layer, you need to use codes below:
    // Dense d; DenseInfo df(...);//remember to fill the DenseInfo construct function
    // add(&d, &df);
    //All three function will automatically call BuildLayer() function in each layer class.
    template <typename T, typename L>
    bool add(T* layer, L* info, const char* optimizer=nullptr)
    {
        if (!(info))
        {
            printf("ERROR: invalid argument info (LayerInfo struct) when calling add() function. the argument maybe NULL.\n");
            getchar();
            return false;
        }
        if (!layer)
        {
            printf("ERROR: invalid argument layer (T*) when calling add() function. the argument may be NULL or not allocated.\n");
            getchar();
            return false;
        }

        if (std::is_same<T, Dense>::value && std::is_same<L, DenseInfo>::value)
        {
            DenseInfo* p = reinterpret_cast<DenseInfo*>(info);
            if (!reinterpret_cast<Dense*>(layer)->BuildLayer(p->input_units, p->output_units, p->activation, optimizer, p->drop_rate))
            {
                printf("Error happened when adding Dense layer.\n");
                getchar();
                return false;
            }
            //record the pointer of layer, how many nodes are in the layer, the type of layer, and the layer's order
            layers.push_back(Net_LayerInfo((void*)layer, p->output_units, DENSE, order));
            order++;
            return true;
        }

        if (std::is_same<T, Flatten>::value && std::is_same<L, FlattenInfo>::value)
        {
            FlattenInfo* p = reinterpret_cast<FlattenInfo*>(info);
            if (!reinterpret_cast<Flatten*>(layer)->BuildLayer(p->input_row, p->input_col))
            {
                printf("Error happened when adding Flatten layer.\n");
                getchar();
                return false;
            }
            layers.push_back(Net_LayerInfo((void*)layer, 1, FLATTEN, order));
            order++;
            return true;
        }

        if (std::is_same<T, Conv2D>::value && std::is_same<L, Conv2DInfo>::value)
        {
            Conv2DInfo* p = reinterpret_cast<Conv2DInfo*>(info);
            if (!reinterpret_cast<Conv2D*>(layer)->BuilderLayer(p->input_rows,p->input_cols,p->kernel_rows,p->kernel_cols,
                p->stride_row, p->stride_col, p->channels, p->layer_units, p->channels/*1*/, p->padding, p->activation))
            {
                printf("Error happend when adding Conv2D layer.\n");
                getchar();
                return false;
            }
            layers.push_back(Net_LayerInfo((void*)layer, p->layer_units, CONV2D, order));
            order++;
            return true;
        }

        if (std::is_same<T, Maxpooling>::value && std::is_same<L, MaxpoolInfo>::value)
        {
            MaxpoolInfo* p = reinterpret_cast<MaxpoolInfo*>(info);
            if (!reinterpret_cast<pMaxp>(layer)->BuildLayer(p->input_units, p->input_rows, p->input_cols, 1, p->kernel_rows,
                p->kernel_cols, p->stride_row, p->stride_col, p->poolmode))
            {
                printf("Error happend when adding Maxpooling layer.\n");
                getchar();
                return false;
            }
            layers.push_back(Net_LayerInfo((void*)layer, p->input_units, MAXPOOLING, order));
            order++;
            return true;
        }

        //You are adding a layer with unknown layer type, the function failed. 
        return false;
    }

    //When you want to add Dense layer, Dropout layer, or Flatten layer, use this function
    template <typename T>
    bool add(T* layer, const int node_num, const MYTYPE drop_rate=0.0, const char* activation="relu", const char* optimizer=nullptr, bool freeze = false)
    {
        if (!layer)
        {
            printf("ERROR: parament layer is null pointer.\n");
            getchar();
            return false;
        }
        if (node_num <= 0)
        {
            printf("ERROR: parament node_num should larger than zero (node_num > 0).\n");
            getchar();
            return false;
        }
        if (layers.empty())
        {
            printf("The first layer of the neural network cannot add new layers by using this function.\nPlease use add(layer, info) function.\n");
            getchar();
            return false;
        }

        if (std::is_same<T, Dense>::value)
        {
            if (layers.back().type == FLATTEN)
            {
                if(!reinterpret_cast<Dense*>(layer)->BuildLayer(reinterpret_cast<Flatten*>(layers.back().layer)->GetSize(), node_num, activation, optimizer, drop_rate, freeze))
                {
                    printf("Error happened when adding Dense layer.\n");
                    getchar();
                    return false;
                }
            }
            if (layers.back().type == DENSE)
            {
                if (!reinterpret_cast<Dense*>(layer)->BuildLayer(layers.back().num, node_num, activation, optimizer, drop_rate, freeze))
                {
                    printf("Error happened when adding Dense layer.\n");
                    getchar();
                    return false;
                }
            }
            if (layers.back().type == DROPOUT)
            {
                if (!reinterpret_cast<Dense*>(layer)->BuildLayer(reinterpret_cast<Dropout*>(layers.back().layer)->Getsize(),
                    node_num, activation, optimizer, drop_rate))
                {
                    printf("Error happened when adding Dense layer.\n");
                    getchar();
                    return false;
                }
            }
            layers.push_back(Net_LayerInfo((void*)layer, node_num, DENSE, order));
            order++;
            return true;
        }

        if (std::is_same<T, Flatten>::value)
        {
            if (layers.back().type == MAXPOOLING)
            {
                if (!reinterpret_cast<Flatten*>(layer)->BuildLayer(reinterpret_cast<pMaxp>(layers.back().layer)->GetOutputRow(),
                    reinterpret_cast<pMaxp>(layers.back().layer)->GetOutputCol()))
                {
                    {
                        printf("Error happened when adding Flatten layer (previous layer is Maxpooling).\n");
                        getchar();
                        return false;
                    }
                }
            }
            if (layers.back().type == CONV2D)
            {
                if (!reinterpret_cast<Flatten*>(layer)->BuildLayer(reinterpret_cast<Conv2D*>(layers.back().layer)->GetOutputRow(),
                    reinterpret_cast<Conv2D*>(layers.back().layer)->GetOutputCol()))
                {
                    {
                        printf("Error happened when adding Flatten layer (previous layer is Conv2D).\n");
                        getchar();
                        return false;
                    }
                }
            }
            layers.push_back(Net_LayerInfo((void*)layer, 1, FLATTEN, order));
            order++;
            return true;
        }

        if (std::is_same<T, Dropout>::value)
        {
            if (layers.back().type == DENSE)
            {
                if (!reinterpret_cast<Dropout*>(layer)->BuildLayer(layers.back().num, drop_rate))
                {
                    printf("Error happened when adding Dropout layer (previous layer is Dense).\n");
                    getchar();
                    return false;
                }
            }

            if (layers.back().type == FLATTEN)
            {
                if (!reinterpret_cast<Dropout*>(layer)->BuildLayer(reinterpret_cast<Flatten*>(layers.back().layer)->GetSize(), drop_rate))
                {
                    printf("Error happened when adding Dropout layer (previous layer is Dense).\n");
                    getchar();
                    return false;
                }
            }
            layers.push_back(Net_LayerInfo((void*)layer, 1, DROPOUT, order));
            order++;
            return true;
        }

        return false;
    }

    //When you want to add Conv2D layer or Maxpooling layer, you can use this function
    template <typename T>
    bool add(T* layer, const int node_num, const int kernel_rows, const int kernel_cols, const int stride_rows, const int stride_cols,
        const char* padding_mode, const char* activation)
    {
        if (!layer)
        {
            printf("ERROR: parament layer is null pointer.\n");
            getchar();
            return false;
        }
        if (node_num <= 0)
        {
            printf("ERROR: parament node_num should larger than zero (node_num > 0).\n");
            getchar();
            return false;
        }
        if (layers.empty() && !std::is_same<T, Flatten>::value)
        {
            printf("Unless this is Flatten layer, the first layer of the neural network cannot add new layers by using this function.\nPlease use add(layer, info) function.\n");
            getchar();
            return false;
        }

        if (std::is_same<T, Conv2D>::value)
        {
            Conv2D* p = nullptr;
            Maxpooling* m = nullptr;
            switch (layers.back().type)
            {
            case CONV2D:
                p = reinterpret_cast<Conv2D*>(layers.back().layer);
                if (!reinterpret_cast<Conv2D*>(layer)->BuilderLayer(p->OutR(), p->OutC(), kernel_rows, kernel_cols,
                    stride_rows, stride_cols, p->GetUnits(), node_num, p->GetUnits(), padding_mode, activation))
                {
                    printf("Error happened when adding Conv2D layer (previous layer is Conv2D).\n");
                    getchar();
                    return false;
                }
                break;
            case MAXPOOLING:
                m = reinterpret_cast<pMaxp>(layers.back().layer);
                if(!reinterpret_cast<Conv2D*>(layer)->BuilderLayer(m->OutRow(),m->OutCol(),kernel_rows,kernel_cols,
                    stride_rows,stride_cols,m->unit(),node_num,m->unit(),padding_mode,activation))
                {
                    printf("Error happened when adding Conv2D layer (previous layer is Maxpooling).\n");
                    getchar();
                    return false;
                }
                break;
            default:
                printf("ERROR: cannot find previous layer's type. Adding this layer failed.\n");
                getchar();
                return false;
            }

            layers.push_back(Net_LayerInfo((void*)layer, node_num, CONV2D, order));
            order++;
            return true;
        }

        if (std::is_same<T, Flatten>::value)
        {
            if (layers.back().type == MAXPOOLING)
            {
                if (!reinterpret_cast<Flatten*>(layer)->BuildLayer(reinterpret_cast<pMaxp>(layers.back().layer)->OutRow(),
                    reinterpret_cast<pMaxp>(layers.back().layer)->OutCol()))
                {
                    {
                        printf("Error happened when adding Flatten layer (previous layer is Maxpooling).\n");
                        getchar();
                        return false;
                    }
                }
            }
            if (layers.back().type == CONV2D)
            {
                if (!reinterpret_cast<Flatten*>(layer)->BuildLayer(reinterpret_cast<Conv2D*>(layers.back().layer)->GetOutputRow(),
                    reinterpret_cast<Conv2D*>(layers.back().layer)->GetOutputRow()))
                {
                    {
                        printf("Error happened when adding Flatten layer (previous layer is Conv2D).\n");
                        getchar();
                        return false;
                    }
                }
            }
            layers.push_back(Net_LayerInfo((void*)layer, 1, FLATTEN, order));
            order++;
            return true;
        }

        if (std::is_same<T, Maxpooling>::value)
        {
            Conv2D* p = nullptr;
            Maxpooling* m = nullptr;
            switch (layers.back().type)
            {
            case CONV2D:
                p = reinterpret_cast<Conv2D*>(layers.back().layer);
                if (!reinterpret_cast<pMaxp>(layer)->BuildLayer(p->GetOutputRow(), p->OutR(), p->OutC(), p->GetChannels(), kernel_rows, kernel_cols,
                    stride_rows, stride_cols))
                {
                    printf("Error happened when adding Conv2D layer (previous layer is Conv2D).\n");
                    getchar();
                    return false;
                }
                break;
            case MAXPOOLING:
                m = reinterpret_cast<pMaxp>(layers.back().layer);
                if(!reinterpret_cast<pMaxp>(layer)->BuildLayer(m->unit(),m->OutRow(),m->OutCol(), m->unit(),kernel_rows,kernel_cols,stride_rows,stride_cols))
                {
                    printf("Error happened when adding Conv2D layer (previous layer is Maxpooling).\n");
                    getchar();
                    return false;
                }
                break;
            default:
                printf("ERROR: cannot find previous layer's type. Adding this layer failed.\n");
                getchar();
                return false;
            }
            
            layers.push_back(Net_LayerInfo((void*)layer, node_num, MAXPOOLING, order));
            order++;
            return true;
        }
        return false;
    }

    int cate_res[10] = { 0 };//array stores how many samples classified into one of the ten categories
    MYTYPE total_loss = 0.0;
    bool train = true;
    int confuse[10][10] = { 0 };
private:
    vector<Net_LayerInfo> layers;//a vector stores all layers' information in the network
    int order = 0;//a variable records the order of the layer, begins from zero.
    MYTYPE loss = 0.0f;//loss value for single sample
    int batch_size = 1;
    int current_sample = 0;
    int current_batch = 1;
    int batches = 0;
    int rest_sample = 0;
    Vector input;
    //MYTYPE* lastLayer_output = nullptr;
    bool update = false;
};
