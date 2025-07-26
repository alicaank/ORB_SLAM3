#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>

// Abstract base class for depth estimation inference
class DepthEstimationInference {
protected:
    std::unique_ptr<Ort::Session> session;
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<Ort::AllocatedStringPtr> input_names_allocated;
    std::vector<Ort::AllocatedStringPtr> output_names_allocated;
    std::vector<int64_t> input_shape;

public:
    DepthEstimationInference(const std::string& model_name);
    virtual ~DepthEstimationInference() = default;

    virtual cv::Mat preprocessImage(const cv::Mat& image) = 0;
    virtual cv::Mat postprocessDepth(const cv::Mat& depth_map, int target_width, int target_height) = 0;
    virtual cv::Mat extractDepthFromOutput(std::vector<Ort::Value>& output_tensors) = 0;

    cv::Mat filterDepthByDistance(const cv::Mat& depth_map, float max_distance_meters);
    cv::Mat filterDepthByRange(const cv::Mat& depth_map, float min_distance_meters, float max_distance_meters);

    void initializeSession(const std::string& model_path, bool use_gpu = true);
    virtual void fixDynamicDimensions();
    void setupInputOutputInfo();
    void printModelInfo();
    std::vector<float> matToVector(const cv::Mat& mat);

    cv::Mat runInference(const cv::Mat& image);
    cv::Mat runInferenceWithFiltering(const cv::Mat& image, float max_distance_meters);
    cv::Mat runInferenceWithRangeFiltering(const cv::Mat& image, float min_distance_meters, float max_distance_meters);
    cv::Mat colorizeDepth(const cv::Mat& depth_map, int colormap = cv::COLORMAP_PLASMA);
    void saveDepthMap(const cv::Mat& depth_map, const std::string& output_path);

    struct PerformanceStats {
        double avg_time_ms;
        double fps;
        double total_time_ms;
    };
    PerformanceStats measurePerformance(const cv::Mat& image, int num_runs = 10, int warmup_runs = 3);
    void printPerformanceStats(const PerformanceStats& stats, int num_runs);
};

// Depth-Anything-V2 implementation
class DepthAnythingV2Inference : public DepthEstimationInference {
private:
    const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> std = {0.229f, 0.224f, 0.225f};
public:
    DepthAnythingV2Inference(const std::string& model_path, bool use_gpu = true);
    cv::Mat preprocessImage(const cv::Mat& image) override;
    cv::Mat extractDepthFromOutput(std::vector<Ort::Value>& output_tensors) override;
    cv::Mat postprocessDepth(const cv::Mat& depth_map, int target_width, int target_height) override;
    cv::Mat infer(const cv::Mat& image);
};

// UniDepth implementation
class UniDepthInference : public DepthEstimationInference {
private:
    std::vector<int64_t> output_shape_pts3d;
    std::vector<int64_t> output_shape_confidence;
    std::vector<int64_t> output_shape_intrinsics;
public:
    struct InferenceResult {
        cv::Mat depth;
        cv::Mat confidence;
        cv::Mat intrinsics;
    };
    UniDepthInference(const std::string& model_path, bool use_gpu = true);
    void fixDynamicDimensions() override;
    void setupUniDepthOutputShapes();
    cv::Mat preprocessImage(const cv::Mat& image) override;
    cv::Mat extractDepthFromOutput(std::vector<Ort::Value>& output_tensors) override;
    cv::Mat postprocessDepth(const cv::Mat& depth_map, int target_width, int target_height) override;
    InferenceResult inferFull(const cv::Mat& image);
    cv::Mat infer(const cv::Mat& image);
};

// Factory function
std::unique_ptr<DepthEstimationInference> createDepthEstimator(const std::string& model_type, 
                                                              const std::string& model_path, 
                                                              bool use_gpu = true);