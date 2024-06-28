#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include "image_processor.h"
#include "common_helper_cv.h"

#define WORK_DIR RESOURCE_DIR
#define LOOP_NUM_FOR_TIME_MEASUREMENT 10
#define FOCAL_LENGTH 1000.0
#define REAL_HEIGHT 1.75
const std::string FRAME_SEPARATOR = "\xFF\xD8\xFF\xD9";

// Reduce the frame rate to capture every nth frame
int32_t FRAME_RATE_REDUCTION = 6;

template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable cond;

public:
    ThreadSafeQueue() {}

    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(std::move(value));
        cond.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this]{ return !queue.empty(); });
        T value = std::move(queue.front());
        queue.pop();
        return value;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }
};

void captureFrames(cv::VideoCapture& cap, ThreadSafeQueue<cv::Mat>& frames) {
    int frame_cnt = 0;
    while (cap.isOpened()) {
        cv::Mat frame;
        if (cap.read(frame)) {
            if (!frame.empty() && frame_cnt++ % FRAME_RATE_REDUCTION == 0) {
                frames.push(frame);
            }
        } else {
            break;
        }
    }
}

void processFrames(ThreadSafeQueue<cv::Mat>& frames) {
    
    while (true) {
        cv::Mat frame = frames.pop();
        if (frame.empty()) break; // Stop if empty frame signals end of video stream
        
        cv::resize(frame, frame, cv::Size(640, 480));

        ImageProcessor::Result result;
        int process_status = ImageProcessor::Process(frame, result, FOCAL_LENGTH, REAL_HEIGHT);
        if (process_status != 0) {
//	      FRAME_RATE_REDUCTION = 8;
//            std::cout<<"yes"<<std::endl;
        } else {
            // Process frame if needed
  //          FRAME_RATE_REDUCTION = 6;
    //        std::cout<<"NO"<<std::endl;
        }
	cv::resize(frame, frame, cv::Size(501, 301));

        std::vector<uchar> buffer;
       cv::imencode(".jpg", frame, buffer);

        // Output the encoded frame
          std::cout.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
          std::cout.write(FRAME_SEPARATOR.data(), FRAME_SEPARATOR.size());
          std::cout.flush();

        // Optionally display frame
//        cv::imshow("Processed Frame", frame);

        if (cv::waitKey(1) == 'q') break;
    }
}

void initializeImageProcessor() {
    ImageProcessor::InputParam input_param = { WORK_DIR, 4 };
    if (ImageProcessor::Initialize(input_param) != 0) {
        std::cerr << "Error initializing ImageProcessor" << std::endl;
        std::exit(-1);
    }
}

void setPrintMemoryLimit() {
    std::setvbuf(stdout, nullptr, _IOFBF, 100 * 1024 * 1024); // Set buffer size to 100 MB
}

int main(int argc, char* argv[]) {
    setPrintMemoryLimit(); // Set the print buffer size to 100 MB

    std::thread initThread(initializeImageProcessor);
    initThread.join();
    cv::VideoCapture cap("rtspsrc location=rtsp://192.168.8.1:8900/live latency=0 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! appsink", cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    ThreadSafeQueue<cv::Mat> frameQueue;
    std::thread producer(captureFrames, std::ref(cap), std::ref(frameQueue));
    std::thread consumer(processFrames, std::ref(frameQueue));

    producer.join();
    consumer.join();

    ImageProcessor::Finalize();
    cv::destroyAllWindows();

    return 0;
}

