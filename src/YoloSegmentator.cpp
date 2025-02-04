#include "YoloSegmentator.hpp"

// #define DEBUG_PRINT

#ifdef DEBUG_PRINT
#define DEBUG(x) std::cout << x << std::endl
#else
#define DEBUG(x)
#endif

namespace yolo
{
    
YoloSegmentator::YoloSegmentator(string& mpath, string model_name) {
    DEBUG("[Constructor] Initializing YoloSegmentator...");
    env = Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "yolov11");
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);  // Use CUDA device 0
    session = new Session(env, mpath.c_str(), session_options);
    DEBUG("[Constructor] YoloSegmentator initialized successfully");
}

YoloSegmentator::~YoloSegmentator() {
    DEBUG("[Destructor] Cleaning up YoloSegmentator");
    delete session;
    DEBUG("[Destructor] Cleanup complete");
}

bool YoloSegmentator::segment(const Mat& img, vector<Obj>& objs){
    DEBUG("[segment] Input image size: " << img.size());
    
    Mat image;
    resize(img, image, Size(NET_W, NET_H));
    DEBUG("[segment] Resized to: " << image.size());
            
    Mat blob = blobFromImage(image, 1 / 255.0, Size(NET_W, NET_H), Scalar(0, 0, 0), true, false);
    DEBUG("[segment] Blob size: " << blob.size());
    //debug input_Shape
    vector<int64_t> input_shape = { 1, 3, NET_W, NET_H };
    Value input_tensor = Value::CreateTensor<float>(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        (float*)blob.data, 3 * NET_W * NET_H, input_shape.data(), input_shape.size());
    DEBUG("[segment] Input tensor created");
    auto start = chrono::high_resolution_clock::now();
    //check if session is null
    if (session == nullptr) {
        DEBUG("[segment] Session is null");
        return false;
    }
    auto output_tensors = session->Run(Ort::RunOptions{ nullptr },
		input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
    // auto output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), 2);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    DEBUG("[segment] Inference time: " << duration.count() << "ms");

    float* all_data = output_tensors[0].GetTensorMutableData<float>();
    auto data_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    Mat output0 = Mat(Size((int)data_shape[2], (int)data_shape[1]), CV_32F, all_data).t();
    auto mask_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    vector<int> mask_sz = { 1,(int)mask_shape[1],(int)mask_shape[2],(int)mask_shape[3] };
    Mat output1 = Mat(mask_sz, CV_32F, output_tensors[1].GetTensorMutableData<float>());
    ImageInfo img_info = { img.size(), { 640.0 / img.cols ,640.0 / img.rows,0,0 } };

    DEBUG("[segment] Output0 shape: " << data_shape[0] << "x" << data_shape[1] << "x" << data_shape[2]);
    DEBUG("[segment] Output1 shape: " << mask_shape[0] << "x" << mask_shape[1] << "x" << mask_shape[2] << "x" << mask_shape[3]);

    decode(output0, output1, img_info, objs);
    
    DEBUG("[segment] Detected " << objs.size() << " objects");
    return objs.size() > 0;
}

void YoloSegmentator::decode(Mat& output0, Mat& output1, ImageInfo para, vector<Obj>& output) {
    DEBUG("[decode] Starting decode with image size: " << para.raw_size);
    output.clear();
    vector<int> class_ids;
    vector<float> accus;
    vector<Rect> boxes;
    vector<vector<float>> masks;
    int data_width = class_names.size() + 4 + 32;
    int rows = output0.rows;
    float* pdata = (float*)output0.data;
    for (int r = 0; r < rows; ++r)
    {
        Mat scores(1, class_names.size(), CV_32FC1, pdata + 4);
        Point class_id;
        double max_socre;
        minMaxLoc(scores, 0, &max_socre, 0, &class_id);
        if (max_socre >= ACCU_THRESH)
        {
            masks.push_back(vector<float>(pdata + 4 + class_names.size(), pdata + data_width));
            float w = pdata[2] / para.trans[0];
            float h = pdata[3] / para.trans[1];
            int left = MAX(int((pdata[0] - para.trans[2]) / para.trans[0] - 0.5 * w + 0.5), 0);
            int top = MAX(int((pdata[1] - para.trans[3]) / para.trans[1] - 0.5 * h + 0.5), 0);
            class_ids.push_back(class_id.x);
            accus.push_back(max_socre);
            boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
        }

        pdata += data_width;//next line
    }
    vector<int> nms_result;
    NMSBoxes(boxes, accus, ACCU_THRESH, MASK_THRESH, nms_result);
    for (int i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        if (class_ids[idx] != 0) //only person
        {
            continue;
        }
        boxes[idx] = boxes[idx] & Rect(0, 0, para.raw_size.width, para.raw_size.height);
        Obj result = { class_ids[idx] ,accus[idx] ,boxes[idx] };
        get_mask(Mat(masks[idx]).t(), output1, para, boxes[idx], result.mask);
        output.push_back(result);
    }
    DEBUG("[decode] Decode complete. Found " << output.size() << " objects");
}

void YoloSegmentator::get_mask(const Mat& mask_info, const Mat& mask_data, const ImageInfo& para, Rect bound, Mat& mast_out)
 {
    Vec4f trans = para.trans;
    int r_x = floor((bound.x * trans[0] + trans[2]) / NET_W * SEG_W);
    int r_y = floor((bound.y * trans[1] + trans[3]) / NET_H * SEG_H);
    int r_w = ceil(((bound.x + bound.width) * trans[0] + trans[2]) / NET_W * SEG_W) - r_x;
    int r_h = ceil(((bound.y + bound.height) * trans[1] + trans[3]) / NET_H * SEG_H) - r_y;
    r_w = MAX(r_w, 1);
    r_h = MAX(r_h, 1);
    if (r_x + r_w > SEG_W) //crop
    {
        SEG_W - r_x > 0 ? r_w = SEG_W - r_x : r_x -= 1;
    }
    if (r_y + r_h > SEG_H)
    {
        SEG_H - r_y > 0 ? r_h = SEG_H - r_y : r_y -= 1;
    }
    vector<Range> roi_rangs = { Range(0, 1) ,Range::all() , Range(r_y, r_h + r_y) ,Range(r_x, r_w + r_x) };
    Mat temp_mask = mask_data(roi_rangs).clone();
    Mat protos = temp_mask.reshape(0, { SEG_CH,r_w * r_h });
    Mat matmul_res = (mask_info * protos).t();
    Mat masks_feature = matmul_res.reshape(1, { r_h,r_w });
    //print mask feature

    Mat dest;
    exp(-masks_feature, dest);//sigmoid
    dest = 1.0 / (1.0 + dest);
    //print mask
    int left = floor((NET_W / SEG_W * r_x - trans[2]) / trans[0]);
    int top = floor((NET_H / SEG_H * r_y - trans[3]) / trans[1]);
    int width = ceil(NET_W / SEG_W * r_w / trans[0]);
    int height = ceil(NET_H / SEG_H * r_h / trans[1]);
    Mat mask;
    resize(dest, mask, Size(width, height));
    mast_out = mask(bound - Point(left, top)) > MASK_THRESH;
}

} // namespace yolo