#include "DepthInference.h"
#include <iostream>
#include <stdexcept>
#include <chrono>

// ---- DepthEstimationInference ----
DepthEstimationInference::DepthEstimationInference(const std::string& model_name)
    : env(ORT_LOGGING_LEVEL_WARNING, model_name.c_str()) {}

cv::Mat DepthEstimationInference::filterDepthByDistance(const cv::Mat& depth_map, float max_distance_meters) {
    cv::Mat filtered_depth = depth_map.clone();
    cv::Mat mask = depth_map > max_distance_meters;
    filtered_depth.setTo(0.0f, mask);
    return filtered_depth;
}

cv::Mat DepthEstimationInference::filterDepthByRange(const cv::Mat& depth_map, float min_distance_meters, float max_distance_meters) {
    cv::Mat filtered_depth = depth_map.clone();
    cv::Mat mask_min = depth_map < min_distance_meters;
    cv::Mat mask_max = depth_map > max_distance_meters;
    cv::Mat combined_mask;
    cv::bitwise_or(mask_min, mask_max, combined_mask);
    filtered_depth.setTo(0.0f, combined_mask);
    return filtered_depth;
}

void DepthEstimationInference::initializeSession(const std::string& model_path, bool use_gpu) {
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    if (use_gpu) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "Using GPU (CUDA) for inference" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "GPU not available, falling back to CPU: " << e.what() << std::endl;
        }
    }
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
    setupInputOutputInfo();
}

void DepthEstimationInference::setupInputOutputInfo() {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session->GetInputNameAllocated(i, allocator);
        input_names.push_back(input_name.get());
        input_names_allocated.push_back(std::move(input_name));
    }
    Ort::TypeInfo input_type_info = session->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape = input_tensor_info.GetShape();
    fixDynamicDimensions();
    size_t num_output_nodes = session->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        output_names.push_back(output_name.get());
        output_names_allocated.push_back(std::move(output_name));
    }
    printModelInfo();
}

void DepthEstimationInference::fixDynamicDimensions() {
    if (input_shape[0] == -1) input_shape[0] = 1;
    if (input_shape[2] == -1) input_shape[2] = 518;
    if (input_shape[3] == -1) input_shape[3] = 518;
}

void DepthEstimationInference::printModelInfo() {
    std::cout << "Model loaded successfully" << std::endl;
    std::cout << "Input shape: [" << input_shape[0] << ", " << input_shape[1] 
              << ", " << input_shape[2] << ", " << input_shape[3] << "]" << std::endl;
    std::cout << "Input names: ";
    for (size_t i = 0; i < input_names.size(); i++) std::cout << input_names[i] << " ";
    std::cout << std::endl << "Output names: ";
    for (size_t i = 0; i < output_names.size(); i++) std::cout << output_names[i] << " ";
    std::cout << std::endl;
}

std::vector<float> DepthEstimationInference::matToVector(const cv::Mat& mat) {
    std::vector<float> data;
    std::vector<cv::Mat> channels(3);
    cv::split(mat, channels);
    for (int c = 0; c < 3; c++) {
        cv::Mat channel = channels[c];
        data.insert(data.end(), (float*)channel.data, (float*)channel.data + channel.total());
    }
    return data;
}

cv::Mat DepthEstimationInference::runInference(const cv::Mat& image) {
    int original_height = image.rows;
    int original_width = image.cols;
    cv::Mat processed_image = preprocessImage(image);
    std::vector<float> input_data = matToVector(processed_image);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), 
        input_shape.data(), input_shape.size());
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, 
                                     input_names.data(), input_tensors.data(), 1,
                                     output_names.data(), output_names.size());
    cv::Mat depth_map = extractDepthFromOutput(output_tensors);
    return postprocessDepth(depth_map, original_width, original_height);
}

cv::Mat DepthEstimationInference::runInferenceWithFiltering(const cv::Mat& image, float max_distance_meters) {
    cv::Mat depth_result = runInference(image);
    return filterDepthByDistance(depth_result, max_distance_meters);
}

cv::Mat DepthEstimationInference::runInferenceWithRangeFiltering(const cv::Mat& image, float min_distance_meters, float max_distance_meters) {
    cv::Mat depth_result = runInference(image);
    return filterDepthByRange(depth_result, min_distance_meters, max_distance_meters);
}

cv::Mat DepthEstimationInference::colorizeDepth(const cv::Mat& depth_map, int colormap) {
    cv::Mat colored, normalized_depth;
    double min_val, max_val;
    cv::minMaxLoc(depth_map, &min_val, &max_val);
    if (max_val == 0.0) {
        normalized_depth = cv::Mat::zeros(depth_map.size(), CV_8U);
    } else {
        depth_map.convertTo(normalized_depth, CV_8U, 255.0 / max_val);
    }
    cv::applyColorMap(normalized_depth, colored, colormap);
    return colored;
}

void DepthEstimationInference::saveDepthMap(const cv::Mat& depth_map, const std::string& output_path) {
    cv::imwrite(output_path, depth_map);
}

DepthEstimationInference::PerformanceStats DepthEstimationInference::measurePerformance(const cv::Mat& image, int num_runs, int warmup_runs) {
    std::cout << "Warming up model..." << std::endl;
    for (int i = 0; i < warmup_runs; i++) runInference(image);
    double total_time = 0.0;
    std::cout << "Running " << num_runs << " inference iterations..." << std::endl;
    for (int i = 0; i < num_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        runInference(image);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double inference_time_ms = duration.count();
        total_time += inference_time_ms;
        std::cout << "Run " << (i + 1) << ": " << inference_time_ms << " ms" << std::endl;
    }
    PerformanceStats stats;
    stats.avg_time_ms = total_time / num_runs;
    stats.fps = 1000.0 / stats.avg_time_ms;
    stats.total_time_ms = total_time;
    return stats;
}

void DepthEstimationInference::printPerformanceStats(const PerformanceStats& stats, int num_runs) {
    std::cout << "\nInference Statistics:" << std::endl;
    std::cout << "Average inference time: " << stats.avg_time_ms << " ms" << std::endl;
    std::cout << "Approximate FPS: " << stats.fps << std::endl;
    std::cout << "Total time for " << num_runs << " runs: " << stats.total_time_ms << " ms" << std::endl;
}

// ---- DepthAnythingV2Inference ----
DepthAnythingV2Inference::DepthAnythingV2Inference(const std::string& model_path, bool use_gpu)
    : DepthEstimationInference("DepthAnythingV2Inference") {
    initializeSession(model_path, use_gpu);
}

cv::Mat DepthAnythingV2Inference::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    cv::cvtColor(image, processed, cv::COLOR_BGR2RGB);
    int target_height = input_shape[2];
    int target_width = input_shape[3];
    cv::resize(processed, processed, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
    processed.convertTo(processed, CV_32F, 1.0/255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);
    for (int c = 0; c < 3; c++) channels[c] = (channels[c] - mean[c]) / std[c];
    cv::merge(channels, processed);
    return processed;
}

cv::Mat DepthAnythingV2Inference::extractDepthFromOutput(std::vector<Ort::Value>& output_tensors) {
    auto actual_output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    float* depth_data = output_tensors[0].GetTensorMutableData<float>();
    cv::Mat depth_map;
    if (actual_output_shape.size() == 4) {
        int out_height = actual_output_shape[2];
        int out_width = actual_output_shape[3];
        depth_map = cv::Mat(out_height, out_width, CV_32F, depth_data);
    } else if (actual_output_shape.size() == 3) {
        int out_height = actual_output_shape[1];
        int out_width = actual_output_shape[2];
        depth_map = cv::Mat(out_height, out_width, CV_32F, depth_data);
    } else {
        throw std::runtime_error("Unexpected output shape dimensions");
    }
    return depth_map.clone();
}

cv::Mat DepthAnythingV2Inference::postprocessDepth(const cv::Mat& depth_map, int target_width, int target_height) {
    cv::Mat processed_depth;
    cv::resize(depth_map, processed_depth, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
    return processed_depth;
}

cv::Mat DepthAnythingV2Inference::infer(const cv::Mat& image) {
    return runInference(image);
}

// ---- UniDepthInference ----
UniDepthInference::UniDepthInference(const std::string& model_path, bool use_gpu)
    : DepthEstimationInference("UniDepthInference") {
    initializeSession(model_path, use_gpu);
    setupUniDepthOutputShapes();
}

void UniDepthInference::fixDynamicDimensions() {
    if (input_shape[0] == -1) input_shape[0] = 1;
    if (input_shape[2] == -1) input_shape[2] = 476;
    if (input_shape[3] == -1) input_shape[3] = 630;
}

void UniDepthInference::setupUniDepthOutputShapes() {
    Ort::TypeInfo output_type_info_0 = session->GetOutputTypeInfo(0);
    auto output_tensor_info_0 = output_type_info_0.GetTensorTypeAndShapeInfo();
    output_shape_pts3d = output_tensor_info_0.GetShape();
    Ort::TypeInfo output_type_info_1 = session->GetOutputTypeInfo(1);
    auto output_tensor_info_1 = output_type_info_1.GetTensorTypeAndShapeInfo();
    output_shape_confidence = output_tensor_info_1.GetShape();
    Ort::TypeInfo output_type_info_2 = session->GetOutputTypeInfo(2);
    auto output_tensor_info_2 = output_type_info_2.GetTensorTypeAndShapeInfo();
    output_shape_intrinsics = output_tensor_info_2.GetShape();
}

cv::Mat UniDepthInference::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    cv::cvtColor(image, processed, cv::COLOR_BGR2RGB);
    int target_height = input_shape[2];
    int target_width = input_shape[3];
    cv::resize(processed, processed, cv::Size(target_width, target_height));
    processed.convertTo(processed, CV_32F, 1.0/255.0);
    return processed;
}

cv::Mat UniDepthInference::extractDepthFromOutput(std::vector<Ort::Value>& output_tensors) {
    auto pts3d_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    float* pts3d_data = output_tensors[0].GetTensorMutableData<float>();
    int height = pts3d_shape[2];
    int width = pts3d_shape[3];
    cv::Mat depth_map(height, width, CV_32F);
    for (int i = 0; i < height * width; i++) {
        depth_map.at<float>(i / width, i % width) = pts3d_data[2 * height * width + i];
    }
    return depth_map;
}

cv::Mat UniDepthInference::postprocessDepth(const cv::Mat& depth_map, int target_width, int target_height) {
    cv::Mat processed_depth;
    cv::resize(depth_map, processed_depth, cv::Size(target_width, target_height));
    return processed_depth;
}

UniDepthInference::InferenceResult UniDepthInference::inferFull(const cv::Mat& image) {
    int original_height = image.rows;
    int original_width = image.cols;
    cv::Mat processed_image = preprocessImage(image);
    std::vector<float> input_data = matToVector(processed_image);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), 
        input_shape.data(), input_shape.size());
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, 
                                     input_names.data(), input_tensors.data(), 1,
                                     output_names.data(), output_names.size());
    InferenceResult result;
    auto pts3d_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    auto confidence_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    float* pts3d_data = output_tensors[0].GetTensorMutableData<float>();
    int height = pts3d_shape[2];
    int width = pts3d_shape[3];
    cv::Mat depth_map(height, width, CV_32F);
    for (int i = 0; i < height * width; i++) {
        depth_map.at<float>(i / width, i % width) = pts3d_data[2 * height * width + i];
    }
    result.depth = postprocessDepth(depth_map, original_width, original_height);
    float* confidence_data = output_tensors[1].GetTensorMutableData<float>();
    cv::Mat confidence_map(height, width, CV_32F, confidence_data);
    result.confidence = confidence_map.clone();
    float* intrinsics_data = output_tensors[2].GetTensorMutableData<float>();
    cv::Mat intrinsics_mat(3, 3, CV_32F, intrinsics_data);
    result.intrinsics = intrinsics_mat.clone();
    return result;
}

cv::Mat UniDepthInference::infer(const cv::Mat& image) {
    return runInference(image);
}

// ---- Factory ----
std::unique_ptr<DepthEstimationInference> createDepthEstimator(const std::string& model_type, 
                                                              const std::string& model_path, 
                                                              bool use_gpu) {
    if (model_type == "depthanything" || model_type == "depthanythingv2") {
        return std::make_unique<DepthAnythingV2Inference>(model_path, use_gpu);
    } else if (model_type == "unidepth") {
        return std::make_unique<UniDepthInference>(model_path, use_gpu);
    } else {
        throw std::invalid_argument("Unknown model type: " + model_type);
    }
}