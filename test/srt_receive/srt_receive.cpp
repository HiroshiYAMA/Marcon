/* MIT License
 *
 *  Copyright (c) 2024 Backcasters.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

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
