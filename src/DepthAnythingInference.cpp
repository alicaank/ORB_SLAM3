#include "DepthAnythingInference.hpp"

//#define DEBUG_PRINT

#ifdef DEBUG_PRINT
#define DEBUG(x) std::cout << x << std::endl
#else
#define DEBUG(x)
#endif

namespace depth {

DepthAnythingInference::DepthAnythingInference(
    const string& modelPath, 
    InferenceDevice device,
    int inferenceHeight, 
    int inferenceWidth
) : device(device), height(inferenceHeight), width(inferenceWidth),
    normalize(true), minDepth(0.0f), maxDepth(20.0f),
    inputNames({"image"}), outputNames({"depth"}) {
    
    DEBUG("[Constructor] Initializing DepthAnythingInference...");
    
    // Validate dimensions
    if (!isMultipleOf14(height) || !isMultipleOf14(width)) {
        throw std::invalid_argument("Height and width must be multiples of 14");
    }
    
    // Initialize ONNX Runtime environment
    env = Env(ORT_LOGGING_LEVEL_WARNING, "depth-anything-inference");
    
    // Get available providers
    auto availableProviders = GetAvailableProviders();
    
    // Check if CUDA provider is available
    bool cudaAvailable = false;
    cout << "Available providers: ";
    for (const auto& provider : availableProviders) {
        cout << provider << " ";
        if (provider == "CUDAExecutionProvider") {
            cudaAvailable = true;
        }
    }
    cout << endl;
    
    // Configure session options
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    sessionOptions.DisableMemPattern();
    
    // Set up device provider
    if (device == InferenceDevice::CUDA && cudaAvailable) {
        cout << "CUDA is available for ONNX Runtime inference." << endl;
        
        // Create CUDA provider options
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = 0;
        cudaOptions.arena_extend_strategy = 0;
        cudaOptions.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;  // 2GB limit
        cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
        
        // Add CUDA provider
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        cout << "CUDA provider has been successfully added to session options." << endl;
    } else {
        if (device == InferenceDevice::CUDA && !cudaAvailable) {
            cout << "CUDA requested but not available. Falling back to CPU." << endl;
        } else {
            cout << "Using CPU for inference." << endl;
        }
    }
    
    // Create session
    try {
        cout << "Loading model: " << modelPath << endl;
        session = new Session(env, modelPath.c_str(), sessionOptions);
        cout << "Model loaded successfully" << endl;
    } catch (const Ort::Exception& e) {
        cout << "ONNX Runtime error: " << e.what() << endl;
        throw;
    }
    
    DEBUG("[Constructor] DepthAnythingInference initialized successfully");
}

DepthAnythingInference::~DepthAnythingInference() {
    DEBUG("[Destructor] Cleaning up DepthAnythingInference");
    delete session;
    DEBUG("[Destructor] Cleanup complete");
}

vector<float> DepthAnythingInference::preprocess(const Mat& image) {
    DEBUG("[preprocess] Input image size: " << image.size());
    
    // Create a copy of the image
    Mat img = image.clone();
    
    // Convert BGR to RGB
    Mat imageRGB;
    cvtColor(img, imageRGB, COLOR_BGR2RGB);
    
    // Convert to float and normalize to [0, 1]
    imageRGB.convertTo(imageRGB, CV_32FC3, 1.0/255.0);
    
    // Resize image for inference
    Mat resizedImage;
    resize(imageRGB, resizedImage, Size(width, height), 0, 0, INTER_CUBIC);
    
    // Normalize with ImageNet mean and std
    Scalar mean(0.485, 0.456, 0.406);
    Scalar std(0.229, 0.224, 0.225);
    
    Mat channels[3];
    split(resizedImage, channels);
    
    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - mean[c]) / std[c];
    }
    
    merge(channels, 3, resizedImage);
    
    // Convert to NCHW format (batch, channels, height, width)
    vector<float> inputTensorValues(1 * 3 * height * width);
    
    // Copy data from OpenCV Mat to the input tensor
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                inputTensorValues[(c * height * width) + (h * width) + w] =
                    resizedImage.at<Vec3f>(h, w)[c];
            }
        }
    }
    
    DEBUG("[preprocess] Preprocessing complete");
    return inputTensorValues;
}

void DepthAnythingInference::postprocess(float* depthData, const vector<int64_t>& outputShape, 
                                       const Size& originalSize, DepthResult& result) {
    DEBUG("[postprocess] Output shape: " << outputShape[0] << "x" << outputShape[1] << "x" << outputShape[2]);
    
    // Get output dimensions
    int outputHeight = static_cast<int>(outputShape[1]);
    int outputWidth = static_cast<int>(outputShape[2]);
    
    // Create depth map from output tensor
    Mat depthMap(outputHeight, outputWidth, CV_32FC1, depthData);
    
    // Save raw depth map
    result.depthMap = depthMap.clone();
    
    // Normalize depth for visualization if requested
    if (normalize) {
        // Normalize depth from (minDepth, maxDepth) to (0, 255)
        Mat depthNormalized;
        subtract(depthMap, minDepth, depthNormalized);
        divide(depthNormalized, (maxDepth - minDepth), depthNormalized);
        multiply(depthNormalized, 255.0, depthNormalized);
        
        // Resize to original dimensions
        Mat depthResized;
        resize(depthNormalized, depthResized, originalSize, 0, 0, INTER_CUBIC);
        
        // Convert to 8-bit for display/saving
        depthResized.convertTo(result.depthNormalized, CV_8UC1);
    }
    
    DEBUG("[postprocess] Postprocessing complete");
}

bool DepthAnythingInference::infer(const Mat& image, DepthResult& result) {
    DEBUG("[infer] Processing image with size: " << image.size());
    
    if (image.empty()) {
        cerr << "Error: Input image is empty" << endl;
        return false;
    }
    
    try {
        // Preprocess the image
        vector<float> inputTensorValues = preprocess(image);
        
        // Set input tensor shape (batch, channels, height, width)
        vector<int64_t> inputShape = {1, 3, static_cast<int64_t>(height), static_cast<int64_t>(width)};
        
        // Create input tensor
        MemoryInfo memoryInfo = MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        Value inputTensor = Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputShape.data(), inputShape.size());
        
        // Run inference
        auto start = chrono::high_resolution_clock::now();
        auto outputTensors = session->Run(
            RunOptions{nullptr}, inputNames.data(), &inputTensor, 1,
            outputNames.data(), 1);
        auto end = chrono::high_resolution_clock::now();
        result.inferenceTime = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        
        DEBUG("[infer] Inference completed in " << result.inferenceTime << " ms");
        
        // Validate output
        if (outputTensors.size() != 1) {
            cerr << "Error: Expected 1 output tensor but got " << outputTensors.size() << endl;
            return false;
        }
        
        // Get output tensor data
        float* depthData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Post-process the output
        postprocess(depthData, outputShape, image.size(), result);
        
        return true;
    } catch (const exception& e) {
        cerr << "Error during inference: " << e.what() << endl;
        return false;
    }
}

} // namespace depth