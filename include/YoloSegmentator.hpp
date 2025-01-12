#ifndef YOLOSEGMENTATOR_H
#define YOLOSEGMENTATOR_H

#include<iostream>
#include<memory>
#include <chrono>
#include <sstream>
#include <vector>
using namespace std;


#include<opencv2/opencv.hpp>
using namespace cv;
using namespace cv::dnn;

#include<onnxruntime_cxx_api.h>
using namespace Ort;

#include "ORBextractor.h"
// struct Obj {
//         int id;
//         float accu;
//         Rect bound;
//         Mat mask;
//         vector<float> mask_cofs;  // Add this member
//     };

namespace yolo{
    

    struct ImageInfo {
        Size raw_size;
        Vec4d trans;
    };

    class YoloSegmentator{
        public:
        vector<string> class_names = { "person", "bicycle", "car", "motorbike", "airplane", "bus", "train", 
        "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", "bench", "bird", 
        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", 
        "kite", "baseball_bat", "baseball_glove", "skateboard", "surfboard", "tennis_racket", 
        "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", 
        "chair", "sofa", "potted_plant", "bed", "dining_table", "toilet", "tv_monitor", 
        "laptop", "mouse", "remote", "keyboard", "cell_phone", "microwave", "oven", 
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
        "teddy_bear", "hair_drier", "toothbrush" };

        YoloSegmentator(string& mpath, string model_name);

        ~YoloSegmentator();

        void get_mask(const Mat& mask_info, const Mat& mask_data, const ImageInfo& para, Rect bound, Mat& mast_out);
        
        void decode(Mat& output0, Mat& output1, ImageInfo para, vector<Obj>& output);

        bool segment(const Mat& img, vector<Obj>& objs);

        private:
            const int SEG_CH = 32;
            const int SEG_W = 160, SEG_H = 160;
            const int NET_W = 640, NET_H = 640;
            const float ACCU_THRESH = 0.25, MASK_THRESH = 0.5;
            vector<const char*> input_names = { "images" };
            vector<const char*> output_names = { "output0","output1" };
            vector<int64_t> input_shape = { 1, 3, 640, 640 };
            Session* session;
            Ort::SessionOptions session_options;
            Env env;
    };

}; // namespace yolo

#endif