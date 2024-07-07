/*
 * Copyright (c) 2024, Backcasters. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "common_utils.h"
#include "opencv_utils.h"
// #include "JpegImage.h"
#include "RingBuffer.h"
#include "RemoteServer.h"

namespace {
constexpr auto double_buffer_metadata = false;
}

class ProcLiveView
{
private:
    // cudaStream_t m_cuda_stream = NULL;

    unsigned int thread_count;

    int image_width;
    int image_height;

    std::string gst_str;
    std::unique_ptr<cv::VideoCapture> video_in;

    bool running;

    StopWatch sw;
    double lap_cur;
    double lap_ave;

    struct st_LiveViewInfo
    {
        uint8_t *bitmap_buf = nullptr;
        int bitmap_width = 0;
        int bitmap_height = 0;
    };

    std::mutex mtx;

    st_LiveViewInfo lv_info;
    std::unique_ptr<RingBufferAsync<st_LiveViewInfo>> lv_info_list;

    std::unique_ptr<RingBufferWithPool<uint8_t>> bitmap_buf_list;

    // Metadata.
    int idx_in;
    int idx_out;

public:
    ProcLiveView(const st_RemoteServer &remote_server, int width, int height, unsigned int th_cnt = 0)
    {
        thread_count = (th_cnt == 0) ? std::thread::hardware_concurrency() : th_cnt;
        image_width = width;
        image_height = height;

        // construct GStreamer pipeline.
        {
            std::ostringstream ss;
            ss << "srtclientsrc uri=srt://";
            ss << (remote_server.is_srt_listener ? "" : remote_server.ip_address);
            ss << ":" << remote_server.srt_port << " ! ";
            ss << "tsdemux ! ";
            ss << "queue ! ";
            ss << "h264parse ! video/x-h264 ! ";
#ifdef GST_NV
            ss << "nvv4l2decoder ! ";
            ss << "nvvideoconvert ! video/x-raw,width=" << image_width << ",height=" << image_height << " ! ";
#else
            ss << "avdec_h264 ! ";
            ss << "videoscale n-threads=" << thread_count << " ! video/x-raw,width=" << image_width << ",height=" << image_height << " ! ";
#endif
            ss << "videoconvert n-threads=" << thread_count << " ! video/x-raw,format=BGR ! ";

            ss << "appsink sync=false";
            gst_str = ss.str();
        }

        video_in = std::make_unique<cv::VideoCapture>(gst_str.c_str(), cv::CAP_GSTREAMER);
        if (!video_in->isOpened()) {
            std::cout << "Can't open : " << gst_str << " ..." << std::endl;
            running = false;
            return;
        }

        running = true;

        lap_cur = 1.0;
        lap_ave = 1.0;

        constexpr auto RING_BUF_SZ = 4;
        lv_info = {};
        lv_info_list = std::make_unique<RingBufferAsync<st_LiveViewInfo>>(RING_BUF_SZ);

        bitmap_buf_list = std::make_unique<RingBufferWithPool<uint8_t>>(RING_BUF_SZ);

        idx_in = 0;
        idx_out = double_buffer_metadata ? 1 : 0;
    }

    virtual ~ProcLiveView() {}

    uint8_t *get_bitmap_buf() const { return lv_info.bitmap_buf; }
    int get_bitmap_width() const { return lv_info.bitmap_width; }
    int get_bitmap_height() const { return lv_info.bitmap_height; }

    bool is_running() const { return running; }
    void start() { running = true; }
    void stop() { running = false; }

    std::tuple<double, double> get_lap() { return { lap_cur, lap_ave }; }

    void fetch(bool latest = false)
    {
        lv_info = latest ? lv_info_list->PeekLatest() : lv_info_list->Peek();
    }

    void next(bool latest = false)
    {
        std::lock_guard<std::mutex> lg(mtx);

        latest ? lv_info_list->ReadLatest() : lv_info_list->Read();

        bitmap_buf_list->pop_read_buf();
    }

    void run()
    {
        while (running)
        {
#ifndef NDEBUG
            printf("live view\n");
#endif

            st_LiveViewInfo lvi = {};

            cv::Mat img;
            {
                *video_in >> img;
                if (img.empty()) {
                    std::cout << "[Video Capture] image is empty." << std::endl;
                    continue;
                }
            }
            lvi.bitmap_buf = img.data;
            lvi.bitmap_width = img.cols;
            lvi.bitmap_height = img.rows;

#ifndef NDEBUG
            printf("bitmap adr:size = 0x%p : %d x %d\n", lvi.bitmap_buf, lvi.bitmap_width, lvi.bitmap_height);
#endif

            size_t bitmap_buf_size = lvi.bitmap_width * lvi.bitmap_height * RGB_CH_NUM * sizeof(decltype(lvi.bitmap_buf[0]));
            {
                // alloc ring buffer.
                bitmap_buf_list->alloc(bitmap_buf_size);
            }
            {
                std::lock_guard<std::mutex> lg(mtx);

                // copy buffer.
                auto bitmap_ptr = bitmap_buf_list->Write(lvi.bitmap_buf, bitmap_buf_size);

                // update buffer address.
                lvi.bitmap_buf = bitmap_ptr;

                // write live view info.
                lv_info_list->Write(lvi);
            }
            std::tie(lap_cur, lap_ave) = sw.lap();
        }
    }
};
