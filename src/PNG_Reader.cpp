#include "PNG_Reader.h"
#include <stdio.h>
#include <math.h>
#include <io.h>


PNGReader::PNGReader()
{
    width = 0, height = 0;
    //pixel = nullptr;
}

PNGReader::PNGReader(_In_ cv::Mat& pic)
{
    if (!pic.data)
    {
        printf("ERROR: empty argument pic.\n");
        //success = 0;
        return;
    }
    if (pic.channels() <= 0)
    {
        //success = 0;
        return;
    }
    width = pic.cols;
    height = pic.rows;
    channels = pic.channels();

    if (channels == 1) // binary image
    {
        pixel = Vector(width * height);

        int i = 0;
        for (cv::MatIterator_<uchar> it = pic.begin<uchar>(); it < pic.end<uchar>(), i < width * height; it++, i++)
        {
            pixel[i] = (MYTYPE)*it;
        }
    }
    else // RGB image
    {
        Matrix img = Matrix(channels, width * height, CPU);
        pixel = Vector(channels * width * height);
        std::vector<cv::Mat> img_channel(3);
        cv::split(pic, img_channel);
        int i = 0;
        for (cv::MatIterator_<uchar> it = img_channel[0].begin<uchar>(); it < img_channel[0].end<uchar>(), i < width * height; it++, i++)
            img(0, i) = (MYTYPE)*it; // blue channel

        i = 0;
        for (cv::MatIterator_<uchar> it = img_channel[1].begin<uchar>(); it < img_channel[1].end<uchar>(), i < width * height; it++, i++)
            img(1, i) = (MYTYPE)*it; // green channel

        i = 0;
        for (cv::MatIterator_<uchar> it = img_channel[2].begin<uchar>(); it < img_channel[2].end<uchar>(), i < width * height; it++, i++)
            img(2, i) = (MYTYPE)*it; // red channel

        memcpy(pixel.GetVec(), img.GetMat(), sizeof(MYTYPE) * channels * width * height);
    }
    //success = 1;
}

PNGReader::~PNGReader()
{

}

bool PNGReader::ReadFile(_In_ const char* read_mode, _In_ const char* _filename)
{
    if (!_filename)
    {
        printf("ERROR: null pointer _filename.\n");
        return false;
    }

    if (!read_mode)
    {
        printf("ERROR: null pointer dataset_name.\n");
        return false;
    }

    cv::Mat file;
    if (mystrcmp("grey", read_mode) == 0)
        file = cv::imread(_filename, 0);
    else
        file = cv::imread(_filename);
    if (file.empty())
    {
        printf("ERROR: cannot open %s.\n", _filename);
        return false;
    }
    width = file.cols, height = file.rows;
    channels = file.channels();

    if (strstr(_filename, "/0/"))
    {
        label = 0;
        goto HERE;
    }
    if (strstr(_filename, "/1/"))
    {
        label = 1;
        goto HERE;
    }
    if (strstr(_filename, "/2/"))
    {
        label = 2;
        goto HERE;
    }
    if (strstr(_filename, "/3/"))
    {
        label = 3;
        goto HERE;
    }
    if (strstr(_filename, "/4/"))
    {
        label = 4;
        goto HERE;
    }
    if (strstr(_filename, "/5/"))
    {
        label = 5;
        goto HERE;
    }
    if (strstr(_filename, "/6/"))
    {
        label = 6;
        goto HERE;
    }
    if (strstr(_filename, "/7/"))
    {
        label = 7;
        goto HERE;
    }
    if (strstr(_filename, "/8/"))
    {
        label = 8;
        goto HERE;
    }
    if (strstr(_filename, "/9/"))
        label = 9;

HERE:
    if (channels == 1) // binary image
    {
        pixel = Vector(width * height);
        int count = 0;
        for (cv::MatIterator_<uchar> it = file.begin<uchar>(); it < file.end<uchar>(), count < width * height; it++, count++)
        {
            pixel[count] = (MYTYPE)*it;
        }
        pixel.DataTransfer(HostToDevice);
    }
    else // RGB image
    {
        Matrix img = Matrix(channels, width * height, CPU);
        pixel = Vector(width * height * channels);
        std::vector<cv::Mat> img_channel(3);
        cv::split(file, img_channel);
        int i = 0;
        for (cv::MatIterator_<uchar> it = img_channel[0].begin<uchar>(); it < img_channel[0].end<uchar>(), i < width * height; it++, i++)
            img(0, i) = (MYTYPE)*it; // blue channel

        i = 0;
        for (cv::MatIterator_<uchar> it = img_channel[1].begin<uchar>(); it < img_channel[1].end<uchar>(), i < width * height; it++, i++)
            img(1, i) = (MYTYPE)*it; // green channel

        i = 0;
        for (cv::MatIterator_<uchar> it = img_channel[2].begin<uchar>(); it < img_channel[2].end<uchar>(), i < width * height; it++, i++)
            img(2, i) = (MYTYPE)*it; // red channel

        memcpy(pixel.GetVec(), img.GetMat(), sizeof(MYTYPE) * channels * width * height);
        pixel.DataTransfer(HostToDevice);
    }
    file.~Mat();
    return true;
}

bool PNGReader::OpenFileInDir(_In_ std::string _dir)
{
    std::string currentPath = "";
    std::string filename = "";
    bool flag = true;
    char ext[_MAX_EXT];
    _finddata_t file_info = { 0 };
    intptr_t handel = 0;

    if (_dir.empty())
    {
        printf("ERROR: empty argument _dir.\n");
        return false;
    }
    if (_access(_dir.c_str(), 0) == -1)//用于判断目录是否存在。如果_access不可用，尝试用access代替
    {
        printf("ERROR: Input directory %s does not exsist\n", _dir.c_str());
        return false;
    }

    currentPath = _dir + "/*";
    handel = _findfirst(currentPath.c_str(), &file_info);
    if (-1 == handel)
    {
        printf("ERROR: Maybe the directory %s is empty\n", currentPath.c_str());
        return (!flag);
    }

    do
    {
        if (strcmp(file_info.name, ".") == 0 || strcmp(file_info.name, "..") == 0)
            continue;

        if (file_info.attrib == _A_SUBDIR)
        {
            if (strcmp(file_info.name, "/0/"))
            {
                label = 0;
                goto ENDLABEL;
            }
            if (strcmp(file_info.name, "/1/"))
            {
                label = 1;
                goto ENDLABEL;
            }
            if (strcmp(file_info.name, "/2/"))
            {
                label = 2;
                goto ENDLABEL;
            }
            if (strcmp(file_info.name, "/3/"))
            {
                label = 3;
                goto ENDLABEL;
            }
            if (strcmp(file_info.name, "/4/"))
            {
                label = 4;
                goto ENDLABEL;
            }
            if (strcmp(file_info.name, "/5/"))
            {
                label = 5;
                goto ENDLABEL;
            }
            if (strcmp(file_info.name, "/6/"))
            {
                label = 6;
                goto ENDLABEL;
            }
            if (strcmp(file_info.name, "/7/"))
            {
                label = 7;
                goto ENDLABEL;
            }
            if (strcmp(file_info.name, "/8/"))
            {
                label = 8;
                goto ENDLABEL;
            }
            if (strcmp(file_info.name, "/9/"))
                label = 9;
            
            ENDLABEL:
            filename = _dir + "/" + file_info.name;
            if (!OpenFileInDir(filename.c_str()))
            {
                printf("ERROR: Something wrong when searching sub-dirctory in %s", filename.c_str());
                flag = false;
                break;
            }
            continue;
        }

        _splitpath(file_info.name, NULL, NULL, NULL, ext);
        if (strcmp(ext, ".png") != 0)
            continue;
        filename = _dir + "/*" + file_info.name;
        if (ReadFile("grey", filename.c_str()))
            flag = true;
    } while (!_findnext(handel, &file_info));
    _findclose(handel);

    return flag;
}

void PNGReader::Shrink(ShrinkMode _mode)
{
    if (channels == 1)
    {
        switch (_mode)
        {
        case ZERO_TO_ONE:
            pixel /= 255.0;
            break;

        case MINUS_ONE_TO_ONE:
            pixel = (pixel - 127.5) / 127.5;
            break;

        default:
            break;
        }
    }
    else
    {
        switch (_mode)
        {
        case ZERO_TO_ONE:
            pixel /= 255.0;
            break;
        case MINUS_ONE_TO_ONE:
            pixel = (pixel - 127.5) / 127.5;
            break;
        default:
            break;
        }
    }
}