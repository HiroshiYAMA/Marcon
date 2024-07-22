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

#include <iostream>
#include <iomanip>
#include <sstream>
#include <list>
#include <numeric>
#include <cmath>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <regex>



// #include <fstream>
// #include <cstdlib>
// #include <cstdint>
// #include <string>

// #include <vector>
// #include <queue>
// #include <mutex>
// #include <atomic>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#include <json.hpp>
using njson = nlohmann::json;

#if defined(USE_EXPERIMENTAL_FS)
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#else
    #include <filesystem>
    namespace fs = std::filesystem;
    #if defined(__APPLE__)
        #include <unistd.h>
    #endif
#endif

#ifdef LIVEVIEW_RGBX
#define RGB_CH_NUM  (4)
#else
#define RGB_CH_NUM  (3)
#endif



namespace {

auto ltrim = [](const std::string &s) -> std::string {
    return std::regex_replace(s, std::regex("^\\s+"), std::string(""));
};

auto rtrim = [](const std::string &s) -> std::string {
    return std::regex_replace(s, std::regex("\\s+$"), std::string(""));
};

auto trim = [](const std::string &s) -> std::string {
    return ltrim(rtrim(s));
};

auto split_string = [](const std::string &s, char c) -> std::vector<std::string> {
    std::vector<std::string> v;

    std::stringstream ss{s + c};
    std::string buf;
    while (std::getline(ss, buf, c)) v.push_back(buf);

    return v;
};

auto get_ext = [](std::string filename) -> std::string {
    auto p = fs::path{filename};
    auto ext = p.extension();
    auto ext_str = ext.generic_string();
    std::transform(ext_str.cbegin(), ext_str.cend(), ext_str.begin(), ::toupper);

    return ext_str;
};

auto conv_fps_ms2ND = [](double fps_ms) -> std::tuple<int, int> {
    int N = 120000;
    int D = std::round(fps_ms * 120);

    return { N, D };
};

auto get_string_from_pair_list = [](const auto &vec, auto idx) -> std::string {
    auto itr = std::find_if(vec.begin(), vec.end(), [&idx](auto &e){ return e.first == idx; });
    std::string str = "---";
    if (itr != vec.end()) {
        auto &[k, v] = *itr;
        str = v;
    }

    return str;
};


}

#ifdef USE_JETSON_UTILS
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/cudaResize.h>

namespace {

auto cuda_resize = [](auto in_img, auto iW, auto iH, auto out_img, auto oW, auto oH, auto filter_mode, float max_value = 255.0f, cudaStream_t stream = NULL) -> void {
    if (!in_img || !out_img) return;
    if ((void *)in_img == (void *)out_img) return;

    using type_in = std::remove_reference_t<decltype(in_img[0])>;
    using type_out = std::remove_reference_t<decltype(out_img[0])>;

    if (iW == oW && iH == oH && imageFormatFromType<type_in>() == imageFormatFromType<type_out>()) {
        cudaMemcpyAsync(out_img, in_img, oW * oH * sizeof(type_out), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaResize(in_img, iW, iH, out_img, oW, oH, filter_mode, false, max_value, stream);
    }
};

}
#endif



/////////////////////////////////////////////////////////////////////
// StopWatch.
/////////////////////////////////////////////////////////////////////

class StopWatch
{
    using clock = std::chrono::steady_clock;
    static constexpr size_t queue_max = 100;
private:
    clock::time_point st, et;
    double ave;
    std::list<double> lap_list;

public:
    StopWatch()
    {
        start();
    }

    virtual ~StopWatch() {}

    clock::time_point start()
    {
        st = clock::now();
        return st;
    }

    clock::time_point stop()
    {
        et = clock::now();
        return et;
    }

    double duration()
    {
        auto dt = et - st;
        auto dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count();
        double dt_ms = dt_ns / 1'000.0f / 1'000.0f;

        lap_list.push_back(dt_ms);
        if (lap_list.size() > queue_max) lap_list.pop_front();

        return dt_ms;
    }

    double lap_ave()
    {
        ave = std::accumulate(lap_list.begin(), lap_list.end(), 0.0) / lap_list.size();
        return ave;
    }

    std::tuple<double, double> lap()
    {
        et = stop();
        auto dt_ms = duration();
        lap_ave();
        st = et;

        return { dt_ms, ave };
    }

};

inline auto print_sw_lap = [](StopWatch &sw, int &cnt, const std::string &str) -> void {
    auto [dt_ms, dt_ave] = sw.lap();
    if (cnt++ > 200) {
#ifdef USE_JETSON_UTILS
        // // cudaStreamSynchronize(NULL);
        // cudaDeviceSynchronize();
#endif
        std::cout << str << dt_ms << "(msec), " << dt_ave << "(msec)." << std::endl;
        cnt = 0;
    }
};



/////////////////////////////////////////////////////////////////////
// TinyTimer.
/////////////////////////////////////////////////////////////////////

class TinyTimer
{
    using clock = std::chrono::steady_clock;
private:
    clock::time_point t_pre;

public:
    TinyTimer()
    {
        t_pre = clock::now();
    }

    virtual ~TinyTimer() {}

    // 1 周期待ち.
    void wait1period(double period /* (msec) */)
    {
        auto t_cur = clock::now();
        auto dt = t_cur - t_pre;
        auto dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count();
        double dt_ms = dt_ns / 1'000.0 / 1'000.0;

        constexpr double th = 1.0;
        const double t_sleep = std::max(period - dt_ms, 0.0) - th;
        const int64_t t = t_sleep;
        if (t > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(t));
        }

        do {
            t_cur = clock::now();
            dt = t_cur - t_pre;
            dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count();
            dt_ms = dt_ns / 1'000.0 / 1'000.0;
        } while (dt_ms < period);

        t_pre = t_cur;
    }
};
