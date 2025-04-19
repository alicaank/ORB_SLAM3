#ifndef DEPTH_ANYTHING_INFERENCE_H
#define DEPTH_ANYTHING_INFERENCE_H

#include <iostream>
#include <memory>
#include <chrono>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

namespace depth {

enum class InferenceDevice {
    CPU,
    CUDA
};

struct DepthResult {
    Mat depthMap;       // Raw depth map (CV_32FC1)
    Mat depthNormalized;  // Normalized depth map (CV_8UC1)
    int inferenceTime;  // Inference time in milliseconds
};

class DepthAnythingInference {
public:
    /**
     * Constructor for DepthAnythingInference
     * 
     * @param modelPath Path to the ONNX model file
     * @param device Inference device (CPU or CUDA)
     * @param inferenceHeight Height for inference (must be multiple of 14)
     * @param inferenceWidth Width for inference (must be multiple of 14)
     */
    DepthAnythingInference(
        const string& modelPath, 
        InferenceDevice device = InferenceDevice::CUDA,
        int inferenceHeight = 476, 
        int inferenceWidth = 644
    );

    /**
     * Destructor
     */
    ~DepthAnythingInference();

    /**
     * Process a single image and generate depth map
     * 
     * @param image Input RGB image
     * @param result Output depth result
     * @return True if successful, false otherwise
     */
    bool infer(const Mat& image, DepthResult& result);

    /**
     * Set whether to normalize depth maps for visualization
     * 
     * @param normalize True to normalize, false to return raw depth values
     */
    void setNormalize(bool normalize) { this->normalize = normalize; }

    /**
     * Set depth normalization range
     * 
     * @param minDepth Minimum depth value (typically 0)
     * @param maxDepth Maximum depth value (typically 20 for Depth Anything models)
     */
    void setDepthRange(float minDepth, float maxDepth) {
        this->minDepth = minDepth;
        this->maxDepth = maxDepth;
    }

private:
    // ONNX Runtime objects
    Env env;
    SessionOptions sessionOptions;
    Session* session;

    // Model parameters
    int height;
    int width;
    bool normalize;
    float minDepth;
    float maxDepth;
    InferenceDevice device;

    // Input and output names
    vector<const char*> inputNames;
    vector<const char*> outputNames;

    /**
     * Check if a number is a multiple of 14 (required for model)
     * 
     * @param value The number to check
     * @return True if value is a multiple of 14 and >= 14
     */
    bool isMultipleOf14(int value) const {
        return value % 14 == 0 && value >= 14;
    }

    /**
     * Preprocess image for inference
     * 
     * @param image Input BGR image
     * @return Preprocessed image tensor data
     */
    vector<float> preprocess(const Mat& image);

    /**
     * Postprocess depth output
     * 
     * @param depthData Pointer to depth data from model output
     * @param outputShape Shape of output tensor
     * @param originalSize Original image size
     * @param result Output depth result
     */
    void postprocess(float* depthData, const vector<int64_t>& outputShape, 
                    const Size& originalSize, DepthResult& result);
};

} // namespace depth

#endif // DEPTH_ANYTHING_INFERENCE_H