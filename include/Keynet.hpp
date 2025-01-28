#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

class KeyNetInference {
public:
    /**
     * @brief Construct a new Key Net Inference object
     * 
     * @param detector_model_path Path to the ONNX detector model
     * @param hynet_model_path Path to the ONNX HyNet descriptor model
     * @param num_levels Number of pyramid levels for multi-scale detection
     * @param scale_factor Scale factor between pyramid levels
     * @param nms_size Size of the NMS window
     * @param patch_size Size of the descriptor patches
     * @param s_mult Scale multiplier for patch extraction
     * @param nms_threshold Threshold for NMS
     * @param batch_size_desc Batch size for descriptor computation
     */
    KeyNetInference(const std::string& detector_model_path, 
                   const std::string& hynet_model_path,
                   int num_levels = 3,
                   float scale_factor = M_SQRT2,
                   int nms_size = 15,
                   int patch_size = 32,
                   float s_mult = 22.0f,
                   float nms_threshold = 1.124f,
                   int batch_size_desc = 1000);

    /**
     * @brief Extract keypoints and descriptors from an image
     * 
     * @param input_image Input image (grayscale or RGB)
     * @param keypoints Output vector to store detected keypoints
     * @param descriptors Output matrix to store computed descriptors
     * @param max_keypoints Maximum number of keypoints to detect
     */
    void extractFeatures(const cv::Mat& input_image, 
                        std::vector<cv::KeyPoint>& keypoints,
                        cv::Mat& descriptors,
                        int max_keypoints = 5000);

private:
    // Configuration parameters matching training specs
    static const int NUM_FILTERS = 8;        // M = 8 filters as per training
    static const int KERNEL_SIZE = 5;        // 5x5 kernel size as per training
    int num_levels;                          // Number of pyramid levels
    float scale_factor;                      // Scale factor between levels
    int nms_size;                           // NMS window size
    int patch_size;                         // Patch size for descriptor
    float s_mult;                           // Scale multiplier
    float nms_threshold;                    // NMS threshold
    int batch_size_desc;                    // Batch size for descriptors
    
    // M-SIP window sizes and loss terms from training
    const std::vector<int> MSIP_WINDOW_SIZES = {8, 16, 24, 32, 40};
    const std::vector<int> MSIP_LOSS_TERMS = {256, 64, 16, 4, 1};

    // ONNX Runtime members
    Ort::Env env;
    std::unique_ptr<Ort::Session> detector_session;
    std::unique_ptr<Ort::Session> hynet_session;
    Ort::MemoryInfo memory_info;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    cv::cuda::GpuMat gpu_buffer;

    // Helper functions
    cv::Mat removeBorders(const cv::Mat& score_map, int borders);
    cv::Mat customPyrDown(const cv::Mat& input, float factor);
    std::vector<cv::KeyPoint> performNMS(const cv::Mat& score_map, int nms_size, float threshold);
    cv::Mat extractDescriptorPatches(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints);
    cv::Mat computeHyNetDescriptors(const cv::Mat& patches);
};
