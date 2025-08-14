#include "DepthInference.h"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cstring>
#include <onnxruntime_cxx_api.h>  
#include <tensorrt_provider_factory.h>
                     // C++ helpers

// ---- DepthEstimationInference ----
DepthEstimationInference::DepthEstimationInference(const std::string& model_name)
    : env(ORT_LOGGING_LEVEL_WARNING, model_name.c_str()), 
      streams_initialized(false), 
      io_binding_initialized(false) {
    // Initialize CUDA streams and events
    initializeCudaStreams();
}

DepthEstimationInference::~DepthEstimationInference() {
    cleanupCudaResources();
}

void DepthEstimationInference::initializeCudaStreams() {
    if (streams_initialized) return;
    
    // Create separate CUDA streams for different operations
    cudaError_t err = cudaStreamCreate(&rgb_stream);
    if (err != cudaSuccess) {
        std::cout << "Warning: Failed to create RGB stream: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaStreamCreate(&depth_stream);
    if (err != cudaSuccess) {
        std::cout << "Warning: Failed to create depth stream: " << cudaGetErrorString(err) << std::endl;
        cudaStreamDestroy(rgb_stream);
        return;
    }
    
    err = cudaStreamCreate(&memcpy_stream);
    if (err != cudaSuccess) {
        std::cout << "Warning: Failed to create memcpy stream: " << cudaGetErrorString(err) << std::endl;
        cudaStreamDestroy(rgb_stream);
        cudaStreamDestroy(depth_stream);
        return;
    }
    
    // Create CUDA events for synchronization
    err = cudaEventCreate(&rgb_complete_event);
    if (err != cudaSuccess) {
        std::cout << "Warning: Failed to create RGB event: " << cudaGetErrorString(err) << std::endl;
        cleanupCudaResources();
        return;
    }
    
    err = cudaEventCreate(&depth_complete_event);
    if (err != cudaSuccess) {
        std::cout << "Warning: Failed to create depth event: " << cudaGetErrorString(err) << std::endl;
        cleanupCudaResources();
        return;
    }
    
    err = cudaEventCreate(&memcpy_complete_event);
    if (err != cudaSuccess) {
        std::cout << "Warning: Failed to create memcpy event: " << cudaGetErrorString(err) << std::endl;
        cleanupCudaResources();
        return;
    }
    
    streams_initialized = true;
    std::cout << "CUDA streams initialized successfully" << std::endl;
}

void DepthEstimationInference::initializeIOBinding() {
    if (!streams_initialized || io_binding_initialized) return;
    
    try {
        if (session) {
            io_binding = Ort::IoBinding(*session);
            io_binding_initialized = true;
            std::cout << "I/O binding initialized successfully" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Warning: Failed to initialize I/O binding: " << e.what() << std::endl;
    }
}

void DepthEstimationInference::synchronizeStreams() {
    if (!streams_initialized) return;
    
    // Record events for each stream
    cudaEventRecord(rgb_complete_event, rgb_stream);
    cudaEventRecord(depth_complete_event, depth_stream);
    cudaEventRecord(memcpy_complete_event, memcpy_stream);
    
    // Synchronize all streams
    cudaEventSynchronize(rgb_complete_event);
    cudaEventSynchronize(depth_complete_event);
    cudaEventSynchronize(memcpy_complete_event);
}

void DepthEstimationInference::cleanupCudaResources() {
    if (streams_initialized) {
        cudaStreamDestroy(rgb_stream);
        cudaStreamDestroy(depth_stream);
        cudaStreamDestroy(memcpy_stream);
        cudaEventDestroy(rgb_complete_event);
        cudaEventDestroy(depth_complete_event);
        cudaEventDestroy(memcpy_complete_event);
        streams_initialized = false;
    }
}

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
            OrtTensorRTProviderOptionsV2* trt_opts = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trt_opts));

                    const char* keys[] = {
                "trt_profile_min_shapes",
                "trt_profile_opt_shapes",
                "trt_profile_max_shapes",
                "trt_fp16_enable",
                "trt_engine_cache_enable",
                "trt_max_workspace_size"
            };
            const char* values[] = {
                /* input name must match the ONNX graph exactly; "input" is common
                for UniDepth‑v2 exports created with torch‑on‑nx. */
            "rgbs:1x3x384x512",   // min
                "rgbs:1x3x384x512",   // opt
                "rgbs:1x3x480x640",   // max  (set larger if you ever feed bigger frames)
                "0",                   // enable FP16 kernels
                "1",                   // cache TensorRT engine under $HOME/.onnxruntime/trt_...
                "2147483648"           // 2 GiB workspace
                        };
            Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(trt_opts, keys, values, 6));

            Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_TensorRT_V2(
                session_options, trt_opts));
            Ort::GetApi().ReleaseTensorRTProviderOptions(trt_opts);


            // Optionally add CUDA as fallback
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

            // Initialize I/O binding after session creation
            initializeIOBinding();

            std::cout << "Using TensorRT (and CUDA fallback) for inference" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "TensorRT not available, falling back to CUDA/CPU: " << e.what() << std::endl;
        }
    }
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
    // Use asynchronous inference if CUDA streams are available
    if (streams_initialized && io_binding_initialized) {
        return runInferenceAsync(image);
    } else {
        // Fallback to synchronous inference
        return runInferenceSync(image);
    }
}

cv::Mat DepthEstimationInference::runInferenceSync(const cv::Mat& image) {
    int original_height = image.rows;
    int original_width = image.cols;

    // Start timing
    auto t_start = std::chrono::high_resolution_clock::now();

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

    // End timing
    auto t_end = std::chrono::high_resolution_clock::now();
    double inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::cout << "[DepthInference] Synchronous inference time: " << inference_time_ms << " ms" << std::endl;

    return postprocessDepth(depth_map, original_width, original_height);
}

cv::Mat DepthEstimationInference::runInferenceAsync(const cv::Mat& image) {
    int original_height = image.rows;
    int original_width = image.cols;

    // Start timing
    auto t_start = std::chrono::high_resolution_clock::now();

    // Step 1: Preprocess image (CPU-bound, but we can overlap with GPU operations)
    cv::Mat processed_image = preprocessImage(image);
    std::vector<float> input_data = matToVector(processed_image);
    
    // Step 2: Allocate GPU memory for input and output
    void* gpu_input_buffer;
    size_t input_size = input_data.size() * sizeof(float);
    cudaMalloc(&gpu_input_buffer, input_size);
    
    // Calculate output size
    size_t output_size = 1;
    for (size_t i = 1; i < input_shape.size(); ++i) { // Skip batch dimension
        output_size *= input_shape[i];
    }
    
    void* gpu_output_buffer;
    cudaMalloc(&gpu_output_buffer, output_size * sizeof(float));
    
    // Step 3: Copy input data to GPU on memcpy stream (overlaps with other operations)
    cudaMemcpyAsync(gpu_input_buffer, input_data.data(), input_size, 
                   cudaMemcpyHostToDevice, memcpy_stream);
    cudaEventRecord(memcpy_complete_event, memcpy_stream);
    
    // Step 4: Create ONNX tensors with GPU memory
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, static_cast<float*>(gpu_input_buffer), input_data.size(), 
        input_shape.data(), input_shape.size());
    
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info, static_cast<float*>(gpu_output_buffer), output_size, 
        nullptr, 0);
    
    // Step 5: Bind input and output tensors
    io_binding.BindInput(input_names[0], input_tensor);
    io_binding.BindOutput(output_names[0], output_tensor);
    
    // Step 6: Wait for input data to be ready, then run inference on depth stream
    cudaStreamWaitEvent(depth_stream, memcpy_complete_event, 0);
    session->Run(Ort::RunOptions{nullptr}, io_binding);
    cudaEventRecord(depth_complete_event, depth_stream);
    
    // Step 7: Copy results back to CPU on memcpy stream (overlaps with postprocessing)
    std::vector<float> output_data(output_size);
    cudaStreamWaitEvent(memcpy_stream, depth_complete_event, 0);
    cudaMemcpyAsync(output_data.data(), gpu_output_buffer, output_size * sizeof(float), 
                   cudaMemcpyDeviceToHost, memcpy_stream);
    
    // Step 8: Extract depth map (CPU-bound, but can run while GPU copy is happening)
    cv::Mat depth_map = extractDepthFromOutput(output_data, input_shape[2], input_shape[3]);
    
    // Step 9: Final synchronization
    cudaStreamSynchronize(memcpy_stream);
    
    // Cleanup GPU memory
    cudaFree(gpu_input_buffer);
    cudaFree(gpu_output_buffer);
    
    // End timing
    auto t_end = std::chrono::high_resolution_clock::now();
    double inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::cout << "[DepthInference] Asynchronous inference time: " << inference_time_ms << " ms" << std::endl;

    return postprocessDepth(depth_map, original_width, original_height);
}

cv::Mat DepthEstimationInference::runInferenceWithOverlapping(const cv::Mat& image) {
    if (!streams_initialized || !io_binding_initialized) {
        std::cout << "Warning: CUDA streams not available, falling back to synchronous inference" << std::endl;
        return runInferenceSync(image);
    }

    int original_height = image.rows;
    int original_width = image.cols;

    // Start timing
    auto t_start = std::chrono::high_resolution_clock::now();

    // Step 1: Preprocess image on CPU (can overlap with GPU operations)
    cv::Mat processed_image = preprocessImage(image);
    std::vector<float> input_data = matToVector(processed_image);
    
    // Step 2: Allocate GPU memory for input and output
    void* gpu_input_buffer;
    void* gpu_output_buffer;
    size_t input_size = input_data.size() * sizeof(float);
    size_t output_size = 1;
    for (size_t i = 1; i < input_shape.size(); ++i) {
        output_size *= input_shape[i];
    }
    
    cudaMalloc(&gpu_input_buffer, input_size);
    cudaMalloc(&gpu_output_buffer, output_size * sizeof(float));
    
    // Step 3: Launch memory copy on memcpy stream (overlaps with other operations)
    cudaMemcpyAsync(gpu_input_buffer, input_data.data(), input_size, 
                   cudaMemcpyHostToDevice, memcpy_stream);
    cudaEventRecord(memcpy_complete_event, memcpy_stream);
    
    // Step 4: Create ONNX tensors with GPU memory
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, static_cast<float*>(gpu_input_buffer), input_data.size(), 
        input_shape.data(), input_shape.size());
    
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info, static_cast<float*>(gpu_output_buffer), output_size, 
        nullptr, 0);
    
    // Step 5: Bind tensors
    io_binding.BindInput(input_names[0], input_tensor);
    io_binding.BindOutput(output_names[0], output_tensor);
    
    // Step 6: Wait for input data, then run inference on depth stream
    cudaStreamWaitEvent(depth_stream, memcpy_complete_event, 0);
    session->Run(Ort::RunOptions{nullptr}, io_binding);
    cudaEventRecord(depth_complete_event, depth_stream);
    
    // Step 7: Launch output copy on memcpy stream (overlaps with postprocessing)
    std::vector<float> output_data(output_size);
    cudaStreamWaitEvent(memcpy_stream, depth_complete_event, 0);
    cudaMemcpyAsync(output_data.data(), gpu_output_buffer, output_size * sizeof(float), 
                   cudaMemcpyDeviceToHost, memcpy_stream);
    
    // Step 8: Extract depth map (CPU-bound, runs while GPU copy is happening)
    cv::Mat depth_map = extractDepthFromOutput(output_data, input_shape[2], input_shape[3]);
    
    // Step 9: Final synchronization
    cudaStreamSynchronize(memcpy_stream);
    
    // Cleanup
    cudaFree(gpu_input_buffer);
    cudaFree(gpu_output_buffer);
    
    // End timing
    auto t_end = std::chrono::high_resolution_clock::now();
    double inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::cout << "[DepthInference] Overlapping inference time: " << inference_time_ms << " ms" << std::endl;

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

void DepthEstimationInference::resetStreams() {
    if (!streams_initialized) return;
    
    // Reset all streams to ensure clean state
    cudaStreamSynchronize(rgb_stream);
    cudaStreamSynchronize(depth_stream);
    cudaStreamSynchronize(memcpy_stream);
}

void DepthEstimationInference::demonstrateOverlappingPerformance(const cv::Mat& image, int num_runs) {
    if (!streams_initialized || !io_binding_initialized) {
        std::cout << "CUDA streams not available for overlapping demonstration" << std::endl;
        return;
    }

    std::cout << "\n=== Overlapping Work and Transfers Performance Demonstration ===" << std::endl;
    
    // Warm up
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 3; i++) {
        runInferenceWithOverlapping(image);
    }
    
    // Test synchronous inference
    std::cout << "\nTesting synchronous inference..." << std::endl;
    auto sync_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        runInferenceSync(image);
    }
    auto sync_end = std::chrono::high_resolution_clock::now();
    double sync_time = std::chrono::duration_cast<std::chrono::milliseconds>(sync_end - sync_start).count();
    
    // Test overlapping inference
    std::cout << "\nTesting overlapping inference..." << std::endl;
    auto overlap_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        runInferenceWithOverlapping(image);
    }
    auto overlap_end = std::chrono::high_resolution_clock::now();
    double overlap_time = std::chrono::duration_cast<std::chrono::milliseconds>(overlap_end - overlap_start).count();
    
    // Calculate performance improvement
    double improvement = ((sync_time - overlap_time) / sync_time) * 100.0;
    
    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Synchronous inference time: " << sync_time << " ms" << std::endl;
    std::cout << "Overlapping inference time: " << overlap_time << " ms" << std::endl;
    std::cout << "Performance improvement: " << improvement << "%" << std::endl;
    std::cout << "Average sync time per run: " << sync_time / num_runs << " ms" << std::endl;
    std::cout << "Average overlap time per run: " << overlap_time / num_runs << " ms" << std::endl;
    
    if (improvement > 0) {
        std::cout << "✓ Overlapping work and transfers provides " << improvement << "% performance improvement!" << std::endl;
    } else {
        std::cout << "⚠ Overlapping did not provide improvement in this case" << std::endl;
    }
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

cv::Mat DepthAnythingV2Inference::extractDepthFromOutput(const std::vector<float>& output_data, int height, int width) {
    cv::Mat depth_map(height, width, CV_32F);
    std::memcpy(depth_map.data, output_data.data(), output_data.size() * sizeof(float));
    return depth_map;
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
    setupInputOutputInfo();         // <-- Add this line!
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
    if (image.empty()) {
        throw std::runtime_error("Input image is empty in preprocessImage!");
    }
    if (input_shape.size() < 4) {
        throw std::runtime_error("input_shape has less than 4 elements!");
    }
    int target_height = input_shape[2];
    int target_width = input_shape[3];
    if (target_height <= 0 || target_width <= 0) {
        throw std::runtime_error("Invalid target size in preprocessImage!");
    }
    cv::Mat processed;
    cv::cvtColor(image, processed, cv::COLOR_BGR2RGB);
    cv::resize(processed, processed, cv::Size(target_width, target_height));
    processed.convertTo(processed, CV_32F, 1.0/255.0);
    return processed;
}

cv::Mat UniDepthInference::extractDepthFromOutput(std::vector<Ort::Value>& output_tensors) {
    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        throw std::runtime_error("No tensor output from ONNX inference!");
    }
    auto pts3d_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "pts3d_shape: ";
    for (auto d : pts3d_shape) std::cout << d << " ";
    std::cout << std::endl;
    if (pts3d_shape.size() != 4 || pts3d_shape[1] != 3) {
        throw std::runtime_error("Unexpected pts_3d output shape!");
    }
    float* pts3d_data = output_tensors[0].GetTensorMutableData<float>();
    int height = pts3d_shape[2];
    int width = pts3d_shape[3];
    cv::Mat depth_map(height, width, CV_32F);
    for (int i = 0; i < height * width; i++) {
        depth_map.at<float>(i / width, i % width) = pts3d_data[2 * height * width + i];
    }
    return depth_map;
}

cv::Mat UniDepthInference::extractDepthFromOutput(const std::vector<float>& output_data, int height, int width) {
    cv::Mat depth_map(height, width, CV_32F);
    // For UniDepth, the output is 3D points, we need the Z coordinate (depth)
    for (int i = 0; i < height * width; i++) {
        depth_map.at<float>(i / width, i % width) = output_data[2 * height * width + i];
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