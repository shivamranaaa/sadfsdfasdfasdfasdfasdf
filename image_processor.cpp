/*** Include Section ***/
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>
#include "common_helper.h"
#include "common_helper_cv.h"
#include "bounding_box.h"
#include "detection_engine.h"
#include "feature_engine.h"
#include "tracker_deepsort.h"
#include "image_processor.h"
#include <cmath>  // For hypot() and other mathematical operations
#include <algorithm>  // For min/max function

/*** Macro Definitions ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)
#define USE_DEEPSORT

/*** Global Variables ***/
std::unique_ptr<DetectionEngine> s_det_engine;
std::unique_ptr<FeatureEngine> s_feature_engine;
#ifdef USE_DEEPSORT
TrackerDeepSort s_tracker(200);
#else
TrackerDeepSort s_tracker(2);
#endif

/*** Helper Functions ***/
static cv::Scalar GetColorForId(int32_t id) {
    static constexpr int32_t kMaxNum = 100;
    static std::vector<cv::Scalar> color_list;
    if (color_list.empty()) {
        std::srand(123);
        for (int32_t i = 0; i < kMaxNum; i++) {
            color_list.push_back(cv::Scalar(std::rand() % 255, std::rand() % 255, std::rand() % 255));
        }
    }
    return color_list[id % kMaxNum];
}

std::string ImageProcessor::calculateDroneVelocityFormatted(const cv::Point& prevCentroid, const cv::Point& currCentroid, const cv::Rect& prevBbox, const cv::Rect& currBbox, int frameWidth, int frameHeight) {
    // Frame center
    cv::Point frameCenter(frameWidth / 2, frameHeight / 2);

    // Displacement from the frame center
    int deltaX = currCentroid.x - frameCenter.x;
    int deltaY = currCentroid.y - frameCenter.y;
    
    //std::cout << deltaX << std::endl;

    // Maximum possible displacement from the center (diagonal / 2)
    double maxDistance = hypot(frameWidth / 2, frameHeight / 2);

    // Calculate velocities
    double rightwardVelocity = (1.0 * deltaX / maxDistance); // Normalize and scale to [0, 2] m/s
    double forwardVelocity = -(1.0 * deltaY / maxDistance); // Normalize and scale to [0, 2] m/s, negative for forward

    // Clamp velocities to the range [-2, 2]
    rightwardVelocity = std::max(-1.0, std::min(1.0, rightwardVelocity));
    forwardVelocity = std::max(-1.0, std::min(1.0, forwardVelocity));

    // Calculate yaw change based on rightward movement
    double yawspeedDegS = 0.0; // Initialize to 0.0
    if (deltaX > 60 || deltaX < -60) {
        yawspeedDegS = std::max(-1.0, std::min(1.0, rightwardVelocity * 15)); // Scale and clamp yaw change
    }

    // Format the output as specified
    std::ostringstream output;
    output << forwardVelocity << " " << rightwardVelocity << " " << -forwardVelocity << " " << yawspeedDegS;  //forward or backward, right and left, up and down, yaw
    return output.str();
}


void saveToTextFile(const std::string& data, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << data;
        file.close();
        //std::cout << "Data saved to " << filename << " successfully." << std::endl;
    } else {
        //std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
    }
}

/*** DrawFps Function ***/
void ImageProcessor::DrawFps(cv::Mat& mat, double time_inference_det, double time_inference_feature, int32_t num_feature, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect) {
    char text[128];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %4.1f, Inference: DET: %4.1f[ms], FEATURE:%3d x %4.1f[ms]", fps, time_inference_det, num_feature, time_inference_feature / num_feature);
    CommonHelper::DrawText(mat, text, pos, font_scale, thickness, color_front, color_back, is_text_on_rect);
}
/*** Main Functionalities ***/
int32_t ImageProcessor::Initialize(const ImageProcessor::InputParam& input_param) {
    if (s_det_engine || s_feature_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }
    s_det_engine.reset(new DetectionEngine(0.4f, 0.2f, 0.5f));
    if (s_det_engine->Initialize(input_param.work_dir, input_param.num_threads) != DetectionEngine::kRetOk) {
        s_det_engine->Finalize();
        s_det_engine.reset();
        return -1;
    }
    s_feature_engine.reset(new FeatureEngine());
    if (s_feature_engine->Initialize(input_param.work_dir, input_param.num_threads) != FeatureEngine::kRetOk) {
        s_feature_engine->Finalize();
        s_feature_engine.reset();
        return -1;
    }
    return 0;
}

int32_t ImageProcessor::Finalize(void) {
    if (!s_det_engine || !s_feature_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }
    if (s_det_engine->Finalize() != DetectionEngine::kRetOk) {
        return -1;
    }
    if (s_feature_engine->Finalize() != FeatureEngine::kRetOk) {
        return -1;
    }
    return 0;
}

int32_t ImageProcessor::Process(cv::Mat& mat, ImageProcessor::Result& result, double focalLength, double realHeight) {
    if (!s_det_engine || !s_feature_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    // Load target ID from file
    std::ifstream id_file("/code_drive/src/kamikaze_main/pj_tflite_track_deepsort/gui_data/txt_files/target_id.txt");
    int target_id = -1;
    bool target_id_present = false;

    if (id_file.is_open()) {
        if (id_file >> target_id) {
            target_id_present = true;
        }
        id_file.close();
    } else {
        PRINT_E("Unable to open target ID file\n");
        return -1;
    }

    // Process the image to detect and compute features
    DetectionEngine::Result det_result;
    if (s_det_engine->Process(mat, det_result) != DetectionEngine::kRetOk) {
        return -1;
    }

    if (det_result.bbox_list.empty()) {
        // No detections found, save default velocity
        saveToTextFile("0.0 0.0 0.0 0.0", "/code_drive/src/kamikaze_main/pj_tflite_track_deepsort/gui_data/txt_files/velocity.txt");
        return 0; // Exit early if no detections
    }

    // Continue processing as before when detections are found
    std::vector<std::vector<float>> feature_list;
    double time_pre_process_feature = 0, time_inference_feature = 0, time_post_process_feature = 0;

    cv::Point prevCentroid, currCentroid;
    cv::Rect prevBbox, currBbox;

    for (const auto& bbox : det_result.bbox_list) {
#ifdef USE_DEEPSORT
        FeatureEngine::Result feature_result;
        if (s_feature_engine->Process(mat, bbox, feature_result) == FeatureEngine::kRetOk) {
            feature_list.push_back(feature_result.feature);
            time_pre_process_feature += feature_result.time_pre_process;
            time_inference_feature += feature_result.time_inference;
            time_post_process_feature += feature_result.time_post_process;
        } else {
            feature_list.push_back(std::vector<float>());
        }
#endif
    }

    s_tracker.Update(det_result.bbox_list, feature_list);
    auto& track_list = s_tracker.GetTrackList();

    for (auto& track : track_list) {
        auto& bbox = track.GetLatestData().bbox;
        if (bbox.score == 0) continue;

        currCentroid = cv::Point(bbox.x + bbox.w / 2, bbox.y + bbox.h / 2);
        currBbox = cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h);

        if (!target_id_present || track.GetId() == target_id) {
            std::string droneVelocityString = calculateDroneVelocityFormatted(prevCentroid, currCentroid, prevBbox, currBbox, mat.cols, mat.rows);

            if (track.GetId() == target_id) {
                saveToTextFile(droneVelocityString, "/code_drive/src/kamikaze_main/pj_tflite_track_deepsort/gui_data/txt_files/velocity.txt");
            } else {
                saveToTextFile("0.0 0.0 0.0 0.0", "/code_drive/src/kamikaze_main/pj_tflite_track_deepsort/gui_data/txt_files/velocity.txt");
            }

            cv::Scalar color = GetColorForId(track.GetId());
            cv::rectangle(mat, currBbox, color, 2);
            cv::circle(mat, currCentroid, 3, color, -1);
            CommonHelper::DrawText(mat, std::to_string(track.GetId()) + ": " + bbox.label, cv::Point(bbox.x, bbox.y - 13), 0.35, 1, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(220, 220, 220));

            prevCentroid = currCentroid;
            prevBbox = currBbox;
        }
    }

    DrawFps(mat, det_result.time_inference, time_inference_feature, feature_list.size(), cv::Point(0, 0), 0.35, 1, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(255, 255, 255), true);

    result.time_pre_process = det_result.time_pre_process;
    result.time_inference = det_result.time_inference;
    result.time_post_process = det_result.time_post_process;
    result.time_pre_process += time_pre_process_feature;
    result.time_inference += time_inference_feature;
    result.time_post_process += time_post_process_feature;
    return 0;
}

