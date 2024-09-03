#include "TXTReader.h"

TXTReader::TXTReader()
{
    data_c1 = nullptr;
    data_c2 = nullptr;
    data_c3 = nullptr;
    fp = nullptr;
}

TXTReader::~TXTReader()
{
    if(data_c1)
    {
        for(int i = 0; i < width; i++)
            if(data_c1[i])
            {
                delete[] data_c1[i];
                data_c1[i] = nullptr;
            }
    
        delete[] data_c1;
        data_c1 = nullptr;
    }

    if(data_c2)
    {
        for(int i = 0; i < width; i++)
            if(data_c2[i])
            {
                delete[] data_c2[i];
                data_c2[i] = nullptr;
            }
  
        delete[] data_c2;
        data_c2 = nullptr;
    }

    if(data_c3)
    {
        for(int i = 0; i < width; i++)
            if(data_c3[i])
            {
                delete[] data_c3[i];
                data_c3[i] = nullptr;
            }

        delete[] data_c3;
        data_c3 = nullptr;
    }
}

bool TXTReader::ReadFile(const char* dataset_name, string& file_name)
{
    if(strlen(dataset_name) == 0 || dataset_name == nullptr)
    {
        printf("ERROR: invalid dataset.\n");
        getchar();
        return false;
    }

    if(file_name.empty())
    {
        printf("ERROR: invalid argument file_name.\n");
        getchar();
        return false;
    }
#ifdef linux
    if(strcasecmp(dataset_name, "mnist") == 0)
    {
        width = 28, height = 28;
        channel = 1;
        goto END_DATASET;
    }
    if(strcasecmp(dataset_name, "fashion_mnist") == 0)
    {
        width = 28, height = 28;
        channel = 1;
        goto END_DATASET;
    }
    if(strcasecmp(dataset_name, "cifar-10") == 0)
    {
        width = 32, height = 32;
        channel = 3;
        goto END_DATASET;
    }
    if(strcasecmp(dataset_name, "cifar-100") == 0)
    {
        width = 32, height = 32;
        channel = 3;
        goto END_DATASET;
    }
#else
    if (stricmp(dataset_name, "mnist") == 0)
    {
        width = 28, height = 28;
        channel = 1;
        goto END_DATASET;
    }
    if (stricmp(dataset_name, "fashion_mnist") == 0)
    {
        width = 28, height = 28;
        channel = 1;
        goto END_DATASET;
    }
    if (stricmp(dataset_name, "cifar-10") == 0)
    {
        width = 32, height = 32;
        channel = 3;
        goto END_DATASET;
    }
    if (stricmp(dataset_name, "cifar-100") == 0)
    {
        width = 32, height = 32;
        channel = 3;
        goto END_DATASET;
    }
#endif
    printf("The dataset's information is not recorded in this programe. Please use TXTReader::ReadCustomDataset function.\n");
    getchar();
    return false;

END_DATASET:
    //data_c1 = new MYTYPE*[4];
    //for (int i = 0; i < 4; i++)
    //    data_c1[i] = new MYTYPE[4];
    //for (int i = 0; i < 4; i++)
    //    for (int j = 0; j < 4; j++)
    //        data_c1[i][j] = 1.0;
    //return true;
    if(channel == 1)
    {
        fp = fopen(file_name.c_str(), "r");
        if(!fp)
        {
            printf("ERROR: read file %s failed.\n", file_name.c_str());
            getchar();
            return false;
        }
        try
        {
            data_c1 = new MYTYPE*[width];
            for(int i = 0; i < width; i++)
                data_c1[i] = new MYTYPE[height];
        }
        catch(const std::bad_alloc& e)
        {
            printf("%s\n",e.what());
            return false;
        }
    

        for(int i = 0; i < width; i++)
            for(int j = 0; j < height; j++)
                fscanf(fp, "%lf", &data_c1[i][j]);

        fclose(fp);
    }

    if(channel == 3)
    {
        fp = fopen(file_name.c_str(), "r");
        if(!fp)
        {
            printf("ERROR: read file %s failed.\n", file_name.c_str());
            getchar();
            return false;
        }
        try
        {
            data_c1 = new MYTYPE*[width];
            for(int i = 0; i < width; i++)
                data_c1[i] = new MYTYPE[height];
        }
        catch(const std::bad_alloc& e)
        {
            printf("%s\n",e.what());
            return false;
        }
        try
        {
            data_c2 = new MYTYPE*[width];
            for(int i = 0; i < width; i++)
                data_c2[i] = new MYTYPE[height];
        }
        catch(const std::bad_alloc& e)
        {
            printf("%s\n",e.what());
            return false;
        }
        try
        {
            data_c3 = new MYTYPE*[width];
            for(int i = 0; i < width; i++)
                data_c3[i] = new MYTYPE[height];
        }
        catch(const std::bad_alloc& e)
        {
            printf("%s\n",e.what());
            return false;
        }

        for(int i = 0; i < width; i++)
            for(int j = 0; j < height; j++)
                fscanf(fp, "%lf", &data_c1[i][j]);

        for(int i = 0; i < width; i++)
            for(int j = 0; j < height; j++)
                fscanf(fp, "%lf", &data_c2[i][j]);

        for(int i = 0; i < width; i++)
            for(int j = 0; j < height; j++)
                fscanf(fp, "%lf", &data_c3[i][j]);

        fclose(fp);
    }

    return true;
}

bool TXTReader::ReadCustomDataset(int width, int height, int channel, string& file_name)
{
    this->width = width;
    this->height = height;
    this->channel = channel;

    fp = fopen(file_name.c_str(), "r");
    if(!fp)
    {
        printf("ERROR: read file %s failed.\n", file_name.c_str());
        getchar();
        return false;
    }

    if(channel == 1)
    {
        try
        {
            data_c1 = new MYTYPE*[width];
            for(int i = 0; i < width; i++)
                data_c1[i] = new MYTYPE[height];
        }
        catch(const std::exception& e)
        {
            printf("%s\n",e.what());
            return false;
        }
            for(int i = 0; i < width; i++)
                for(int j = 0; j < height; j++)
                fscanf(fp, "%lf", &data_c1[i][j]);

        fclose(fp);
    }

    return true;
}

void TXTReader::Shrink(ShrinkMode _mode)
{
    switch(_mode)
    {
    case ZERO_TO_ONE:
        if(channel == 1)
        {
            for(int i = 0; i < width; i++)
                for(int j = 0; j < height; j++)
                    data_c1[i][j] /= 255.0;
        }
        if(channel == 3)
        {
            for(int i = 0; i < width; i++)
                for(int j = 0; j < height; j++)
                {
                    data_c1[i][j] /= 255.0;
                    data_c2[i][j] /= 255.0;
                    data_c3[i][j] /= 255.0;
                }
        }
        break;

    case MINUS_ONE_TO_ONE:
        if(channel == 1)
        {
            for(int i = 0; i < width; i++)
                for(int j = 0; j < height; j++)
                    data_c1[i][j] = (data_c1[i][j] - 127.5) / 127.5;
        }
        if(channel == 3)
        {
            for(int i = 0; i < width; i++)
                for(int j = 0; j < height; j++)
                {
                    data_c1[i][j] = (data_c1[i][j] - 127.5) / 127.5;
                    data_c2[i][j] = (data_c2[i][j] - 127.5) / 127.5;
                    data_c3[i][j] = (data_c3[i][j] - 127.5) / 127.5;
                }
        }
        break;
        
    default:
        if(channel == 1)
        {
            for(int i = 0; i < width; i++)
                for(int j = 0; j < height; j++)
                    data_c1[i][j] /= 255.0;
        }
        if(channel == 3)
        {
            for(int i = 0; i < width; i++)
                for(int j = 0; j < height; j++)
                {
                    data_c1[i][j] /= 255.0;
                    data_c2[i][j] /= 255.0;
                    data_c3[i][j] /= 255.0;
                }
        }
    }
    // if(_mode == ZERO_TO_ONE)
    // {
    //     if(channel == 1)
    //     {
    //         for(int i = 0; i < width; i++)
    //             for(int j = 0; j < height; j++)
    //                 data_c1[i][j] /= 255.0f;
    //     }
    //     if(channel == 3)
    //     {
    //         for(int i = 0; i < width; i++)
    //             for(int j = 0; j < height; j++)
    //             {
    //                 data_c1[i][j] /= 255.0f;
    //                 data_c2[i][j] /= 255.0f;
    //                 data_c3[i][j] /= 255.0f;
    //             }
    //     }
    //     return;
    // }
    
    // if(_mode == MINUS_ONE_TO_ONE)
    // {
    //     if(channel == 1)
    //     {
    //         for(int i = 0; i < width; i++)
    //             for(int j = 0; j < height; j++)
    //                 data_c1[i][j] = (data_c1[i][j] - 127.5f) / 127.5f;
    //     }
    //     if(channel == 3)
    //     {
    //         for(int i = 0; i < width; i++)
    //             for(int j = 0; j < height; j++)
    //             {
    //                 data_c1[i][j] = (data_c1[i][j] - 127.5f) / 127.5f;
    //                 data_c2[i][j] = (data_c2[i][j] - 127.5f) / 127.5f;
    //                 data_c3[i][j] = (data_c3[i][j] - 127.5f) / 127.5f;
    //             }
    //     }
    // }
    
}

bool TXTReader::empty()
{
    if(channel == 3)
        if(!data_c1 && !data_c2 && !data_c3)
            return true;
    if(channel == 1)
        if(!data_c1)
            return true;
    if(width == 0 || height== 0 || channel == 0)
        return true;
    return false;
}

bool TXTReader::isSame(int w, int h)
{
    return w == this->width && h == this->height;
}

MYTYPE** TXTReader::Getdata(char which) const
{
    switch (which)
    {
    case 0:
        return data_c1;
        break;
    case 1:
        return data_c2;
        break;
    case 2:
        return data_c3;
        break;
    default:
        return nullptr;
    }
}