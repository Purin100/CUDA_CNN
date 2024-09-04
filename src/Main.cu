#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Matrix.cuh"
#include "utils.h"
#include "Activation.cuh"
#include "Dense.cuh"
#include "Net.h"
#include "TXTReader.h"
#include "im2col.h"
#include "optimizer.h"
#include "PNG_Reader.h"
#include "Pooling.cuh"
#include <stdio.h>
#if defined(linux) || defined(Liunx)
#include <dirent.h>
#endif

#define CLOSEFILE(fp){if(fp) fclose(fp);}
void Listdir(std::string _dir, const char* ext, std::vector<string>& result, bool fulldir = true);

struct Dataset
{
    Dataset(PNGReader* _file, int _label)
    {
        file = _file, label = _label;
    }
    PNGReader* file;
    int label;
};

int main(int argc, char** argv)
{
    PNGReader* file, * valid;
    

    int trainFile_num, testFile_num;
    std::vector<string> trainfiles, testfiles;

//set your dataset path
#ifdef linux
    Listdir(std::string("./trainsamples"), ".png", trainfiles);
    Listdir(std::string("./testsamples"), ".png", testfiles);
#else
     /*Listdir(std::string("./mnist_train"), ".png",  trainfiles);
     Listdir(std::string("./mnist_test"), ".png", testfiles);*/
    Listdir(std::string("./fashion_train"), ".png", trainfiles);
    Listdir(std::string("./fashion_test"), ".png", testfiles);
#endif


#ifdef _DEBUG
    trainFile_num = 10000, testFile_num = 1000;
#else
    trainFile_num = 200, testFile_num = 200;
#endif
    int obj_label;

    int* train_label_arr = nullptr, * test_label_arr = nullptr;
    int epoch = 10;//total training epoch
    int now_epoch = 0;

    //read training data and test data from files
    trainFile_num = trainfiles.size();
    testFile_num = testfiles.size();

    file = new PNGReader[trainFile_num];
    valid = new PNGReader[testFile_num];

    for (int i = 0; i < trainFile_num; i++)
    {
        file[i].ReadFile("grey", trainfiles[i].c_str());
        file[i].Shrink(ZERO_TO_ONE);
    }
    for (int i = 0; i < testFile_num; i++)
    {
        valid[i].ReadFile("grey", testfiles[i].c_str());
        valid[i].Shrink(ZERO_TO_ONE);
    }
    printf("images loaded.\n");

    train_label_arr = new int[trainFile_num];
    test_label_arr = new int[testFile_num];
    for (int i = 0; i < trainFile_num; i++)
        train_label_arr[i] = file[i].label;
    for (int i = 0; i < testFile_num; i++)
        test_label_arr[i] = valid[i].label;

    printf("label loaded.\n");


    printf("File load complete.\n");

    std::vector<Dataset> trainset, testset;
    for (int i = 0; i < trainFile_num; i++)
        trainset.push_back(Dataset(&file[i], train_label_arr[i]));
    for (int i = 0; i < testFile_num; i++)
        testset.push_back(Dataset(&valid[i], test_label_arr[i]));

    trainFile_num = trainset.size();
    testFile_num = testset.size();
    if (argc == 2)
        epoch = atoi(argv[1]);

    int classnum = 10;
    //Declear layer objects
    Dense *d, *d1, *d2, *df;
    d2 = new Dense;
    df = new Dense;
    Flatten flatten;
    FlattenInfo fi(28, 28);
    cudaError_t cudaStatus;

    //one-hot label matrix
    Matrix one_hot = Identity(classnum);

    Conv2D *c1, c2, c3;
    c1 = new Conv2D;
    Maxpooling p1;

    Conv2DInfo c1Info(28, 28, 3, 3, 2, 2, 16, 1, "relu1");

    //add layers to the network
    Net net(trainFile_num, 100);
    net.add(c1, &c1Info);
    net.add(&c2, 16, 3, 3, 2, 2, "valid", "relu1");
    net.add(&p1, 16, 2, 2, 2, 2, "valid", nullptr);
    net.add(&flatten, 1);
    net.add(d2, 32, 1.0, "relu1");
    net.add(df, classnum, 1.0, "linear");

    int count = 0;
    MYTYPE* loss = new MYTYPE[epoch];
    MYTYPE* accuracy = new MYTYPE[epoch]{ 0.0 };
    MYTYPE* tloss = new MYTYPE[epoch];
    FILE* f;

    
    float partition = 0.8f;//partition for train/valid
    int kkk = trainFile_num * partition;
    //shuffle training set
    //std::random_shuffle(trainset.begin(), trainset.end());
    std::random_device dddd;
    std::mt19937_64 mt199(dddd());
    std::shuffle(trainset.begin(), trainset.end(), mt199);

    //Matrix ori1, ori2;
    //Matrix af1, af2;

    //d2.ExportWeight(&ori1);
    //df.ExportWeight(&ori2);
    //main loop
    while (now_epoch < epoch)
    {
        count = 0;
        //input one file each time
        while (count < kkk)//80% of the training files use for training
        {
            //printf("Epoch:%d File: %d\n", now_epoch, count);
            //obj_label = train_label_sub[count];
            //net.Forward(train_sub[count], flatten.GetSize());

            obj_label = trainset[count].label;
            net.Forward(trainset[count].file->pixel,28,28);
            net.Backward(one_hot.RowSlice(obj_label));
            //if(count%100==0)
            //    net.Save(std::to_string(count/100));
            count++;
        }
        printf("Loss in %d epoch: %f\n", now_epoch, net.total_loss / kkk);
        loss[now_epoch] = net.total_loss / kkk;


        //loss[now_epoch] = net.total_loss / trainFile_num;

        //decrease the learning rate in the network
        //if (now_epoch % 5 == 0)
        //    net.lrDecay(now_epoch);

        //reset total_loss, this variable will record loss in validation process
        net.total_loss = 0.0;

        //validation process
        while(count < trainFile_num)//20% of the training files use for validation
        //count ^= count;
        //while(count<testFile_num)
        {
            obj_label = trainset[count].label;
            net.Forward(trainset[count].file->pixel,28,28);
            //obj_label = testset[count].label;
            //net.Forward(testset[count].file);
            if (net.Eval(obj_label, one_hot.RowSlice(obj_label)))
                accuracy[now_epoch] += 1.0;
            count++;
        }

        //save weights every epoch
        if (now_epoch % 1 == 0)
            net.Save(std::to_string(now_epoch));
        accuracy[now_epoch] /= (trainFile_num * (1.0f - partition));
        tloss[now_epoch] = net.total_loss / (trainFile_num * (1.0f - partition));
        //accuracy[now_epoch] /= testFile_num;
        //tloss[now_epoch] = net.total_loss / testFile_num;
        printf("Valid accuracy in %d epoch: %f, loss in this epoch: %f\n", now_epoch, accuracy[now_epoch], tloss[now_epoch]);
        //f = fopen(string(std::to_string(now_epoch) + "/cate_res.txt").c_str(), "w");
        //for (int i = 0; i < 10; i++)
        //    fprintf(f, "%d ", net.cate_res[i]);
        //fclose(f);
        for (int i = 0; i < 10; i++)
            printf("%d ", net.cate_res[i]);
        printf("\n");
        memset(net.cate_res, 0, sizeof(int) * 10);

        net.total_loss = 0.0;
        net.epochReset();
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
                net.confuse[i][j] = 0;
        now_epoch++;
    }

    FILE* facc = fopen("./valid accuracy fashion normal.txt", "w");
    for (int i = 0; i < epoch; i++)
        fprintf(facc, "%f\n", accuracy[i]);
    fclose(facc);
    facc = fopen("./train loss fashion normal.txt", "w");
    for (int i = 0; i < epoch; i++)
        fprintf(facc, "%f\n", loss[i]);
    fclose(facc);
    facc = fopen("./valid loss fashion normal.txt", "w");
    for (int i = 0; i < epoch; i++)
        fprintf(facc, "%f\n", tloss[i]);
    fclose(facc);

    //test process
    count = 0;
    accuracy[0] = 0.0;
    net.total_loss = 0.0;
    while (count < testFile_num)
    {
        obj_label = test_label_arr[count];
        net.Forward(testset[count].file->pixel,28,28);
        if (net.Eval(obj_label, one_hot.RowSlice(obj_label)))
            accuracy[0] += 1.0;
        count++;
    }
    accuracy[0] /= testFile_num;
    printf("Test accuracy in %d epoch: %f, loss in this epoch: %f\n", now_epoch, accuracy[0], net.total_loss / testFile_num);
    /*f = fopen("./cate_res.txt", "w");
    for (int i = 0; i < 10; i++)
        fprintf(f, "%d ", net.cate_res[i]);
    fclose(f);*/
    memset(net.cate_res, 0, sizeof(int) * 10);

    //Destory cuda handle before reset the device
    mc.Destory();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    //free memory
    RELEASE(c1);
    RELEASE(loss);
    RELEASE(tloss);
    RELEASE(file);
    RELEASE(valid);
    RELEASE(train_label_arr);
    RELEASE(test_label_arr);
    RELEASE(accuracy);
    return 0;
}

#if defined(liunx) || defined(Linux)
void Listdir(std::string _dir, const char* ext, std::vector<string>& result)
{
    DIR* dp;
    struct dirent* dirp;
    std::string temp;
    int ext_len, len;
    if (_dir.empty())
    {
        printf("ERROR: empty directory %s\n", _dir.c_str());
        getchar();
        return;
    }

    if (!(dp = opendir(_dir.c_str())))
    {
        perror("opendir");
        return;
    }

    ext_len = strlen(ext);

    if (ext_len == 0)
    {
        while ((dirp = readdir(dp)) != nullptr)
        {
            if (dirp->d_type == DT_DIR)
                result.push_back(dirp->d_name);
        }

    }
    else
    {
        while ((dirp = readdir(dp)) != nullptr)
        {
            if (strcmp(dirp->d_name, ".") == 0 || strcmp(dirp->d_name, "..") == 0)
                continue;

            if (dirp->d_type == DT_DIR)
            {
                temp = _dir + "/" + std::string(dirp->d_name);
                Listdir(temp, ext, result);
            }

            if (strcmp(ext, ".*") == 0)
            {
                //result.push_back(_dir + "/" + std::string(dirp->d_name));
                result.push_back(dirp->d_name);
            }
            else
            {
                len = strlen(dirp->d_name);
                if (strcasecmp(dirp->d_name + (len - ext_len), ext) == 0)
                    result.push_back(_dir + "/" + dirp->d_name);
                else
                    continue;
            }
        }
    }

    closedir(dp);
}
#endif/*Linux*/
#if defined(_WIN64) || defined(_WIN32)
void Listdir(std::string _dir, const char* ext, std::vector<string>& result, bool fulldir)
{
    _finddata_t file_info = { 0 };
    intptr_t handel = 0;
    string currentPath = "";
    string temppath = "";
    char _ext[_MAX_EXT];//后缀名

    if (_dir.empty())
    {
        printf("ERROR: Invalid argument _dir (null).\n");
        return;
    }
    if (_access(_dir.c_str(), 0) == -1)//用于判断目录是否存在。如果_access不可用，尝试用access代替
    {
        printf("ERROR: Input directory %s does not exsist\n", _dir.c_str());
        return;
    }

    currentPath = _dir + "/*";
    handel = _findfirst(currentPath.c_str(), &file_info);
    if (-1 == handel)
    {
        printf("ERROR: Maybe the directory %s is empty\n", currentPath.c_str());
        return;
    }

    if (ext == NULL)
    {
        do
        {
            if (file_info.attrib & _A_SUBDIR)
            {
                result.push_back(file_info.name);
            }
        } while (!_findnext(handel, &file_info));
        _findclose(handel);
    }
    else
    {
        do
        {
            if (strcmp(file_info.name, ".") == 0 || strcmp(file_info.name, "..") == 0)
                continue;
            if (file_info.attrib & _A_SUBDIR)
            {
                temppath = _dir + "/" + file_info.name;
                Listdir(temppath, ext, result, fulldir);
                continue;
            }

            _splitpath(file_info.name, NULL, NULL, NULL, _ext);
            if (strcmp(ext, _ext) != 0)
                continue;
            else
            {
                if (fulldir)
                    result.push_back(_dir + "/" + file_info.name);
                else
                    result.push_back(file_info.name);
            }

        } while (!_findnext(handel, &file_info));
        _findclose(handel);
    }
}
#endif/*Windows*/
