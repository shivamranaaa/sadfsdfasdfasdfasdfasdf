#ifndef IMAGE_PROCESSOR_H_
#define IMAGE_PROCESSOR_H_

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <opencv2/opencv.hpp>

namespace ImageProcessor {

struct InputParam {
    char work_dir[256];
    int32_t num_threads;
};

struct Result {
    double time_pre_process;   // [msec]
    double time_inference;     // [msec]
    double time_post_process;  // [msec]
};

int32_t Initialize(const InputParam& input_param);
int32_t Process(cv::Mat& mat, Result& result, double focalLength, double realHeight);
int32_t Finalize(void);
int32_t Command(int32_t cmd);

std::string calculateDroneVelocityFormatted(const cv::Point& prevCentroid, const cv::Point& currCentroid, const cv::Rect& prevBbox, const cv::Rect& currBbox, int frameWidth, int frameHeight);

void DrawFps(cv::Mat& mat, double time_inference_det, double time_inference_feature, int32_t num_feature, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect = true);

} // namespace ImageProcessor

#endif // IMAGE_PROCESSOR_H_
