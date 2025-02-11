#include "Keynet.hpp"
struct TimingStats {
    float grayscale_conversion = 0;
    float pyramid_creation = 0;
    float detector_inference = 0;
    float nms_processing = 0;
    float keypoint_sorting = 0;
    float patch_extraction = 0;
    float descriptor_inference = 0;
    std::vector<float> level_times;
    
    void print() {
        std::cout << "\n=== Runtime Analysis ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Image Preprocessing:" << std::endl;
        std::cout << "  Grayscale Conversion: " << std::setw(8) << grayscale_conversion << " ms" << std::endl;
        std::cout << "  Pyramid Creation:     " << std::setw(8) << pyramid_creation << " ms" << std::endl;
        
        std::cout << "\nKeypoint Detection:" << std::endl;
        for(size_t i = 0; i < level_times.size(); i++) {
            std::cout << "  Level " << i << " Processing:   " << std::setw(8) << level_times[i] << " ms" << std::endl;
        }
        std::cout << "  Detector Inference:   " << std::setw(8) << detector_inference << " ms" << std::endl;
        std::cout << "  NMS Processing:       " << std::setw(8) << nms_processing << " ms" << std::endl;
        std::cout << "  Keypoint Sorting:     " << std::setw(8) << keypoint_sorting << " ms" << std::endl;
        
        std::cout << "\nDescriptor Computation:" << std::endl;
        std::cout << "  Patch Extraction:     " << std::setw(8) << patch_extraction << " ms" << std::endl;
        std::cout << "  Descriptor Inference: " << std::setw(8) << descriptor_inference << " ms" << std::endl;
        
        float total = grayscale_conversion + pyramid_creation + detector_inference + 
                        nms_processing + keypoint_sorting + patch_extraction + descriptor_inference;
        std::cout << "\nTotal Time:            " << std::setw(8) << total << " ms" << std::endl;
        std::cout << "========================" << std::endl;
    }
} ;

KeyNetInference::KeyNetInference(const std::string& detector_model_path, 
                               const std::string& hynet_model_path,
                               int num_levels,
                               float scale_factor,
                               int nms_size,
                               int patch_size,
                               float s_mult,
                               float nms_threshold,
                               int batch_size_desc)
    : num_levels(num_levels)
    , scale_factor(scale_factor)
    , nms_size(nms_size)
    , patch_size(patch_size)
    , s_mult(s_mult)
    , nms_threshold(nms_threshold)
    , batch_size_desc(batch_size_desc)
    , env(ORT_LOGGING_LEVEL_WARNING, "KeyNetInference")
    , memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUOutput))
    , input_names{"input"}
    , output_names{"output"} {
    
    // Initialize ONNX sessions
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);  // Use CUDA device 0
    
    detector_session = std::make_unique<Ort::Session>(env, detector_model_path.c_str(), session_options);
    hynet_session = std::make_unique<Ort::Session>(env, hynet_model_path.c_str(), session_options);
}

cv::Mat KeyNetInference::removeBorders(const cv::Mat& score_map, int borders) {
    cv::Mat result = score_map.clone();
    result.rowRange(0, borders) = 0;
    result.rowRange(result.rows - borders, result.rows) = 0;
    result.colRange(0, borders) = 0;
    result.colRange(result.cols - borders, result.cols) = 0;
    return result;
}

cv::Mat KeyNetInference::customPyrDown(const cv::Mat& input, float factor) {
    cv::Size new_size(cvRound(input.cols / factor), cvRound(input.rows / factor));
    cv::Mat resized;
    cv::Mat blurred;
    cv::GaussianBlur(input, blurred, cv::Size(5,5), 0);
    cv::resize(blurred, resized, new_size, 0, 0, cv::INTER_LINEAR);
    return resized;
}

std::vector<cv::KeyPoint> KeyNetInference::performNMS(const cv::Mat& score_map, int nms_size, float threshold) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat dilated;
    cv::dilate(score_map, dilated, cv::Mat(), cv::Point(-1, -1), 1);
        
    // Compute local maxima
    cv::Mat maxima = (score_map == dilated); // Boolean mask where score == local max

    // Apply thresholding: Keep only maxima that are greater than threshold
    cv::Mat mask = (score_map > threshold) & maxima;


    for(int i = nms_size; i < score_map.rows - nms_size; i++) {
        for(int j = nms_size; j < score_map.cols - nms_size; j++) {
            float score = score_map.at<float>(i, j);
            if(mask.at<uchar>(i, j)) {
                cv::KeyPoint kp;
                kp.pt.x = j;
                kp.pt.y = i;
                kp.response = score;
                kp.octave = 0;
                keypoints.emplace_back(kp);
            }
        }
    }
    return keypoints;
}

void KeyNetInference::extractFeatures(const cv::Mat& input_image,
                                    std::vector<cv::KeyPoint>& keypoints,
                                    cv::Mat& descriptors,
                                    int max_keypoints) {
    if(input_image.empty()) {
        keypoints.clear();
        descriptors = cv::Mat();
        return;
    }

    TimingStats timing;
    auto t1 = std::chrono::high_resolution_clock::now();
    // Convert to grayscale if needed
    cv::Mat gray_image;
    if(input_image.channels() == 3) {
        cv::cvtColor(input_image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = input_image.clone();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    timing.grayscale_conversion += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    
    // Create image pyramid
    auto t3 = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> pyramid;
    pyramid.reserve(num_levels);
    pyramid.push_back(gray_image);
    
    for(int i = 1; i < num_levels; i++) {
        pyramid.push_back(customPyrDown(pyramid.back(), scale_factor));
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    timing.pyramid_creation += std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
    

    keypoints.clear();
    keypoints.reserve(max_keypoints * num_levels);
 

    // Process each pyramid level

    for(int level = 0; level < num_levels; level++) {
        auto t5 = std::chrono::high_resolution_clock::now();
        cv::Mat current_image = pyramid[level];
        float current_scale = std::pow(scale_factor, level);
        
        // Normalize image
        cv::Mat float_image;
        current_image.convertTo(float_image, CV_32F, 1.0/255.0);
        
        // Prepare input tensor
        std::vector<float> input_tensor_values(float_image.total());
        memcpy(input_tensor_values.data(), float_image.ptr<float>(), float_image.total() * sizeof(float));
        
        std::vector<int64_t> input_dims = {1, 1, 
            static_cast<int64_t>(float_image.rows), 
            static_cast<int64_t>(float_image.cols)};
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_dims.data(), input_dims.size());
        
        // Run detector inference
        auto output_tensors = detector_session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            1);
        auto t6 = std::chrono::high_resolution_clock::now();
        timing.detector_inference += std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();
        
        // Get score map
        auto t7 = std::chrono::high_resolution_clock::now();
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        cv::Mat score_map(float_image.size(), CV_32F, output_data);
        // Remove borders and perform NMS
        score_map = removeBorders(score_map, nms_size);
        std::vector<cv::KeyPoint> level_keypoints = performNMS(score_map, nms_size, nms_threshold);
        auto t8 = std::chrono::high_resolution_clock::now();
        timing.nms_processing += std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count();
        
        // Scale keypoints back to original image size
        for(auto& kp : level_keypoints) {
            kp.pt.x *= current_scale;
            kp.pt.y *= current_scale;
            kp.size = patch_size * current_scale;
            kp.octave = level;
        }

        keypoints.insert(keypoints.end(), level_keypoints.begin(), level_keypoints.end());
    }
    
    // Sort keypoints by score and limit to max_keypoints
    auto t9 = std::chrono::high_resolution_clock::now();
    if(keypoints.size() > max_keypoints) {
        std::sort(keypoints.begin(), keypoints.end(),
                 [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                     return a.response > b.response;
                 });
        keypoints.resize(max_keypoints);
    }
    auto t10 = std::chrono::high_resolution_clock::now();
    timing.keypoint_sorting = std::chrono::duration_cast<std::chrono::milliseconds>(t10 - t9).count();

    // timing.print(); 
}

cv::Mat KeyNetInference::extractDescriptorPatches(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints) {
    if(keypoints.empty()) return cv::Mat();
    
    std::vector<cv::Mat> patches;
    patches.reserve(keypoints.size());
    
    // Convert image to grayscale for HyNet
    cv::Mat gray_image;
    if(image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image.clone();
    }
    
    // Normalize image to [0, 1] as per training
    cv::Mat float_image;
    gray_image.convertTo(float_image, CV_32F, 1.0/255.0);
    
    // Create transformation matrices for each M-SIP window size
    std::vector<cv::Mat> transforms;
    transforms.reserve(MSIP_WINDOW_SIZES.size());
    
    for(const auto& kp : keypoints) {
        float scale = kp.size * s_mult;
        cv::Point2f center = kp.pt;
        
        // Apply affine transformation considering M-SIP window sizes
        cv::Mat M = cv::getRotationMatrix2D(center, kp.angle, scale);
        M.at<double>(0,2) += patch_size/2 - center.x;
        M.at<double>(1,2) += patch_size/2 - center.y;
        
        cv::Mat patch;
        cv::warpAffine(float_image, patch, M, cv::Size(patch_size, patch_size), 
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        
        // Ensure patch has correct size and type
        if(patch.size() != cv::Size(patch_size, patch_size)) {
            cv::resize(patch, patch, cv::Size(patch_size, patch_size));
        }
        
        patches.push_back(patch);
    }
    
    // Stack patches into a single tensor
    cv::Mat stacked_patches;
    if(!patches.empty()) {
        cv::vconcat(patches, stacked_patches);
        stacked_patches = stacked_patches.reshape(1, patches.size()); // Reshape to [N, 1, H, W]
    }
    
    return stacked_patches;
}

cv::Mat KeyNetInference::computeHyNetDescriptors(const cv::Mat& patches) {
    if(patches.empty()) return cv::Mat();
    
    cv::Mat descriptors;
    int total_patches = patches.rows;
    
    // Process in batches
    for(int i = 0; i < total_patches; i += batch_size_desc) {
        int batch_end = std::min(i + batch_size_desc, total_patches);
        int current_batch_size = batch_end - i;
        
        cv::Mat batch = patches.rowRange(i, batch_end);
        
        // Convert to float and normalize if not already
        cv::Mat float_batch;
        if(batch.type() != CV_32F) {
            batch.convertTo(float_batch, CV_32F, 1.0/255.0);
        } else {
            float_batch = batch;
        }
        
        // Prepare input tensor for current batch
        std::vector<int64_t> input_dims = {
            static_cast<int64_t>(current_batch_size), // batch size
            1,                                        // channels
            static_cast<int64_t>(patch_size),         // height
            static_cast<int64_t>(patch_size)          // width
        };
        
        std::vector<float> input_tensor_values(float_batch.total());
        memcpy(input_tensor_values.data(), float_batch.ptr<float>(), float_batch.total() * sizeof(float));
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_dims.data(), input_dims.size());
        
        // Run HyNet inference for current batch
        auto output_tensors = hynet_session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            1);
        
        // Get output data and clone immediately to own the memory
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Convert to cv::Mat and concatenate
        cv::Mat batch_descriptors(current_batch_size, output_shape[1], CV_32F, output_data);
        cv::Mat batch_descriptors_clone = batch_descriptors.clone();  // Clone to own the data
        
        // L2 normalize descriptors as per training
        for(int r = 0; r < batch_descriptors_clone.rows; r++) {
            cv::Mat row = batch_descriptors_clone.row(r);
            double norm = cv::norm(row);
            if(norm > std::numeric_limits<float>::epsilon()) {
                row /= norm;
            }
        }
        
        if(descriptors.empty()) {
            descriptors = batch_descriptors_clone;
        } else {
            cv::vconcat(descriptors, batch_descriptors_clone, descriptors);
        }
    }
    
    return descriptors;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <path_to_detector_model> <path_to_hynet_model> <path_to_image>" << std::endl;
        return 1;
    }

    std::string detector_model_path = "/home/ak/GuidedResearch/ORB_SLAM3/models/keynet.onnx";
    std::string hynet_model_path = "/home/ak/GuidedResearch/ORB_SLAM3/models/hynet.onnx";
    std::string image_path = "/home/ak/GuidedResearch/Key.Net-Pytorch/test.png";
    std::cout << "CUDA devices: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
    }
    try {
        // Initialize KeyNet inference
        KeyNetInference keynet(detector_model_path, hynet_model_path);
        
        // Load and process image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return 1;
        }
        
        // Run inference
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        //time
        for(int i = 0; i < 10; i++) {

        auto start = std::chrono::high_resolution_clock::now();
        keynet.extractFeatures(image, keypoints, descriptors);
        auto end = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Time taken: " << time << " ms" << std::endl;
        }
        
        // Display results
        cv::Mat output_image = image.clone();
        for(const auto& kp : keypoints) {
            cv::circle(output_image, kp.pt, 2, cv::Scalar(0, 255, 0), -1);
        }
        cv::imshow("Output", output_image);
        cv::waitKey(0);
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
