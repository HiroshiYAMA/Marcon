#include <iostream>

#include <opencv2/opencv.hpp>

int main(int ac, char *av[])
{
    std::ostringstream ss;

//    ss << "srtserversrc uri=srt://:4201 ! ";
    ss << "srtclientsrc uri=srt://43.30.217.166:4201 ! ";
    ss << "tsdemux ! ";
    ss << "queue ! ";
    ss << "h264parse ! video/x-h264 ! ";
    ss << "nvv4l2decoder ! ";
    ss << "nvvideoconvert ! videoconvert ! video/x-raw,format=BGR ! ";
    ss << "appsink sync=false";
    std::string gst_str = ss.str();

    cv::VideoCapture video_in(gst_str.c_str(), cv::CAP_GSTREAMER);
    if (!video_in.isOpened()) {
        std::cout << "Can't open : " << gst_str << " ..." << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat img;
    while (true) {
        video_in >> img;
        if (img.empty()) {
            std::cout << "image is empty." << std::endl;
            continue;
        }
        cv::imshow("VideoCapture", img);

        // escで終了
        auto key = cv::waitKey(2);
        if (key == '\x1b') break;
    }

    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
