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

#include <opencv2/opencv.hpp>

#ifdef GAPI_ENABLE
    #include <opencv2/gapi.hpp>
    #include <opencv2/gapi/core.hpp>
    #include <opencv2/gapi/imgproc.hpp>
    #if CV_VERSION_MAJOR >= 4
        // #include <opencv2/gapi/ocl/core.hpp>
        // #include <opencv2/gapi/ocl/imgproc.hpp>
        // #include <opencv2/gapi/ocl/goclkernel.hpp>
        #include <opencv2/gapi/gpu/core.hpp>
        #include <opencv2/gapi/gpu/imgproc.hpp>
        #include <opencv2/gapi/gpu/ggpukernel.hpp>
    #endif
#endif

#ifdef CUDA_ENABLE
    #include <opencv2/cudaarithm.hpp>
    #include <opencv2/cudabgsegm.hpp>
    #include <opencv2/cudacodec.hpp>
    #include <opencv2/cudafeatures2d.hpp>
    #include <opencv2/cudafilters.hpp>
    #include <opencv2/cudaimgproc.hpp>
    // #include <opencv2/cudalegacy.hpp>
    #include <opencv2/cudaobjdetect.hpp>
    #include <opencv2/cudaoptflow.hpp>
    #include <opencv2/cudastereo.hpp>
    #include <opencv2/cudawarping.hpp>
#endif

#ifdef USE_COMPLETION_LIB
    #include <opencv2/ximgproc/edge_filter.hpp>
    #include "completion_export.h"
#endif

namespace {

static std::string get_cv_depth_str(int depth)
{
    std::string str;

    switch (depth) {
    case CV_8U: str = "CV_8U"; break;
    case CV_8S: str = "CV_8S"; break;
    case CV_16U: str = "CV_16U"; break;
    case CV_16S: str = "CV_16S"; break;
    case CV_32S: str = "CV_32S"; break;
    case CV_32F: str = "CV_32F"; break;
    case CV_64F: str = "CV_64F"; break;
#if CV_VERSION_MAJOR >= 4
    case CV_16F: str = "CV_16F"; break;
#endif
    default:
        str = "Unkown";
        break;
    }

    return str;
}

static void print_cvMat(cv::Mat &m)
{
    // 行数
    std::cout << "rows:" << m.rows << std::endl;
    // 列数
    std::cout << "cols:" << m.cols << std::endl;
    // 次元数
    std::cout << "dims:" << m.dims << std::endl;
    // サイズ（2次元の場合）
    std::cout << "size[]:" << m.size().width << "," << m.size().height << std::endl;
    // ビット深度ID
    std::cout << "depth:" << get_cv_depth_str(m.depth()) << std::endl;
    // チャンネル数
    std::cout << "channels:" << m.channels() << std::endl;
    // （複数チャンネルから成る）1要素のサイズ [バイト単位]
    std::cout << "elemSize:" << m.elemSize() << "[byte]" << std::endl;
    // 1要素内の1チャンネル分のサイズ [バイト単位]
    std::cout << "elemSize1 (elemSize/channels):" << m.elemSize1() << "[byte]" << std::endl;
    // 要素の総数
    std::cout << "total:" << m.total() << std::endl;
    // ステップ数 [バイト単位]
    std::cout << "step:" << m.step << "[byte]" << std::endl;
    // 1ステップ内のチャンネル総数
    std::cout << "step1 (step/elemSize1):" << m.step1() << std::endl;
    // データは連続か？
    std::cout << "isContinuous:" << (m.isContinuous() ? "true" : "false") << std::endl;
    // 部分行列か？
    std::cout << "isSubmatrix:" << (m.isSubmatrix() ? "true" : "false") << std::endl;
    // データは空か？
    std::cout << "empty:" << (m.empty() ? "true" : "false") << std::endl;
}
static void print_cvGpuMat(cv::cuda::GpuMat &m)
{
    // 行数
    std::cout << "rows:" << m.rows << std::endl;
    // 列数
    std::cout << "cols:" << m.cols << std::endl;
    // // 次元数
    // std::cout << "dims:" << m.dims << std::endl;
    // サイズ（2次元の場合）
    std::cout << "size[]:" << m.size().width << "," << m.size().height << std::endl;
    // ビット深度ID
    std::cout << "depth:" << get_cv_depth_str(m.depth()) << std::endl;
    // チャンネル数
    std::cout << "channels:" << m.channels() << std::endl;
    // （複数チャンネルから成る）1要素のサイズ [バイト単位]
    std::cout << "elemSize:" << m.elemSize() << "[byte]" << std::endl;
    // 1要素内の1チャンネル分のサイズ [バイト単位]
    std::cout << "elemSize1 (elemSize/channels):" << m.elemSize1() << "[byte]" << std::endl;
    // // 要素の総数
    // std::cout << "total:" << m.total() << std::endl;
    // ステップ数 [バイト単位]
    std::cout << "step:" << m.step << "[byte]" << std::endl;
    // 1ステップ内のチャンネル総数
    std::cout << "step1 (step/elemSize1):" << m.step1() << std::endl;
    // データは連続か？
    std::cout << "isContinuous:" << (m.isContinuous() ? "true" : "false") << std::endl;
    // // 部分行列か？
    // std::cout << "isSubmatrix:" << (m.isSubmatrix() ? "true" : "false") << std::endl;
    // データは空か？
    std::cout << "empty:" << (m.empty() ? "true" : "false") << std::endl;
}

template<typename T>
float calc_value_max(const T &img)
{
    float value_max;

    switch (img.depth()) {
    case CV_8U: case CV_8S: case CV_16U: case CV_16S: case CV_32S:
        value_max = (size_t(1) << (img.elemSize1() * 8)) - 1.0f;
        break;
    case CV_32F: case CV_64F:
#if CV_VERSION_MAJOR >= 4
    case CV_16F:
#endif
        value_max = 1.0f;
        break;
    default:
        value_max = 255.0f;
        break;
    }

    return value_max;
}

auto make_cvmat_3ch = [](const cv::Mat &img) -> cv::Mat {
    cv::Mat img_3ch(img.size(), CV_MAKETYPE(img.depth(), 3));

    int ch = img.channels();
    switch (ch) {
    case 1:
        cv::merge(std::vector<cv::Mat>{ img, img, img }, img_3ch);
        break;
    case 2:
        {
            constexpr int fromTo[] = { 0,0, 0,1, 0,2 };
            cv::mixChannels(img, img_3ch, fromTo, 3);
        }
        break;
    case 3:
        img_3ch = img;
        break;
    case 4:
        {
            constexpr int fromTo[] = { 0,0, 1,1, 2,2 };
            cv::mixChannels(img, img_3ch, fromTo, 3);
        }
        break;
    default:
        break;
    }

    return img_3ch;
};

#if CV_VERSION_MAJOR < 4
cv::Mat blend_image_3ch(const cv::Mat &img_fg_3ch, const cv::Mat &img_bg_3ch, const cv::Mat &img_alpha)
{
    if (img_fg_3ch.depth() != CV_32F || img_fg_3ch.channels() != 3
        || img_bg_3ch.depth() != CV_32F || img_bg_3ch.channels() != 3
        || img_alpha.depth() != CV_32F || img_alpha.channels() != 1
        || img_fg_3ch.size() != img_bg_3ch.size()
        || img_fg_3ch.size() != img_alpha.size()
    ) {
        return cv::Mat{};
    }

    cv::Mat a;
    cv::merge(std::vector<cv::Mat>{ img_alpha, img_alpha, img_alpha }, a);
    cv::Mat img_alpha_inv = 1 - img_alpha;
    cv::Mat a_inv;
    cv::merge(std::vector<cv::Mat>{ img_alpha_inv, img_alpha_inv, img_alpha_inv }, a_inv);
    cv::Mat img_blend_3ch = img_fg_3ch.mul(a) + img_bg_3ch.mul(a_inv);

    return img_blend_3ch;
}

void blend_image(const cv::Mat &img_fg, const cv::Mat &img_bg, const cv::Mat &img_alpha, cv::Mat &img_blend)
{
    // resize.
    cv::Size blend_size(img_blend.size().width, img_blend.size().height);
    cv::Mat img_fg_resized;
    cv::Mat img_bg_resized;
    cv::Mat img_alpha_resized;
    auto inter_type = [](const cv::Mat &in_img, const cv::Mat &out_img) -> cv::InterpolationFlags {
        return ((in_img.size().width < out_img.size().width) && (in_img.size().height < out_img.size().height)
            ? cv::INTER_NEAREST
            : cv::INTER_AREA);
    };
    cv::resize(img_fg, img_fg_resized, blend_size, 0.0, 0.0, inter_type(img_fg, img_blend));
    cv::resize(img_bg, img_bg_resized, blend_size, 0.0, 0.0, inter_type(img_bg, img_blend));
    cv::resize(img_alpha, img_alpha_resized, blend_size, 0.0, 0.0, inter_type(img_alpha, img_blend));

    // 3ch.
    cv::Mat img_fg_3ch = make_cvmat_3ch(img_fg_resized);
    cv::Mat img_bg_3ch = make_cvmat_3ch(img_bg_resized);

    // CV_32F.
    cv::Mat img_fg_32fc3;
    cv::Mat img_bg_32fc3;
    cv::Mat img_alpha_32fc1;
    img_fg_3ch.convertTo(img_fg_32fc3, CV_32F, 1 / calc_value_max(img_fg));
    img_bg_3ch.convertTo(img_bg_32fc3, CV_32F, 1 / calc_value_max(img_bg));
    img_alpha_resized.convertTo(img_alpha_32fc1, CV_32F, 1 / calc_value_max(img_alpha));

    // blend.
    cv::Mat img_blend_32fc3 = blend_image_3ch(img_fg_32fc3, img_bg_32fc3, img_alpha_32fc1);

    // blend channels.
    cv::Mat a_ones = cv::Mat::ones(blend_size, CV_32FC1);
    cv::Mat img_blend_tmp;
    std::vector<cv::Mat> planes_blend;
    int blend_ch = img_blend.channels();
    if (blend_ch == 1) {
        cv::cvtColor(img_blend_32fc3, img_blend_tmp, cv::COLOR_RGB2GRAY, blend_ch);
    } else if (blend_ch == 2) {
        cv::cvtColor(img_blend_32fc3, img_blend_tmp, cv::COLOR_RGB2GRAY, 1);
        cv::merge(std::vector<cv::Mat>{ img_blend_tmp, a_ones }, img_blend_tmp);
    } else if (blend_ch == 3) {
        img_blend_tmp = img_blend_32fc3;
    } else if (blend_ch == 4) {
        cv::split(img_blend_32fc3, planes_blend);
        cv::merge(std::vector<cv::Mat>{ planes_blend[0], planes_blend[1], planes_blend[2], a_ones }, img_blend_tmp);
    } else {
        img_blend_tmp = img_blend_32fc3;
    }

    // blend CV type.
    int blend_depth = img_blend.depth();
    int cv_type = CV_MAKETYPE(blend_depth, blend_ch);
    float blend_max = calc_value_max(img_blend);
    img_blend_tmp.convertTo(img_blend, cv_type, blend_max);
}
#else
// G-API version.
void blend_image(const cv::Mat &img_fg, const cv::Mat &img_bg, const cv::Mat &img_alpha, cv::Mat &img_blend)
{
    cv::GMat in_fg;
    cv::GMat in_bg;
    cv::GMat in_a;

    // resize.
    cv::Size blend_size(img_blend.size().width, img_blend.size().height);
    auto inter_type = [](const cv::Mat &in_img, const cv::Mat &out_img) -> cv::InterpolationFlags {
        return ((in_img.size().width < out_img.size().width) && (in_img.size().height < out_img.size().height)
            ? cv::INTER_NEAREST
            : cv::INTER_AREA);
    };
    cv::GMat img_fg_resized = cv::gapi::resize(in_fg, blend_size, 0.0, 0.0, inter_type(img_fg, img_blend));
    cv::GMat img_bg_resized = cv::gapi::resize(in_bg, blend_size, 0.0, 0.0, inter_type(img_bg, img_blend));
    cv::GMat img_alpha_resized = cv::gapi::resize(in_a, blend_size, 0.0, 0.0, inter_type(img_alpha, img_blend));

    // CV_8U.
    cv::GMat img_fg_8u = cv::gapi::convertTo(img_fg_resized, CV_8U, 255 / calc_value_max(img_fg));
    cv::GMat img_bg_8u = cv::gapi::convertTo(img_bg_resized, CV_8U, 255 / calc_value_max(img_bg));

    // CV_32F.
    cv::GMat img_alpha_32f = cv::gapi::convertTo(img_alpha_resized, CV_32F, 1 / calc_value_max(img_alpha));

    // split.
    auto split_img = [](const cv::GMat &img, int ch)
        -> std::tuple<cv::GMat, cv::GMat, cv::GMat, cv::GMat>
    {
        cv::GMat r, g, b, a;

        switch (ch) {
        case 1:
            r = img;
            g = img;
            b = img;
            a = cv::gapi::addC(255, (g ^ g));   // all 255.
            break;
        case 4:
            std::tie(r, g, b, a) = cv::gapi::split4(img);
            break;
        case 3:
        default:
            std::tie(r, g, b) = cv::gapi::split3(img);
            a = cv::gapi::addC(255, (g ^ g));   // all 255.
            break;
        }

        return { r, g, b, a };
    };
    auto [fg_r, fg_g, fg_b, fg_a] = split_img(img_fg_8u, img_fg.channels());
    auto [bg_r, bg_g, bg_b, bg_a] = split_img(img_bg_8u, img_bg.channels());

    // blend.
    cv::GMat fg_r_32f = cv::gapi::convertTo(fg_r, CV_32F);
    cv::GMat fg_g_32f = cv::gapi::convertTo(fg_g, CV_32F);
    cv::GMat fg_b_32f = cv::gapi::convertTo(fg_b, CV_32F);
    cv::GMat fg_a_32f = cv::gapi::convertTo(fg_a, CV_32F);
    cv::GMat bg_r_32f = cv::gapi::convertTo(bg_r, CV_32F);
    cv::GMat bg_g_32f = cv::gapi::convertTo(bg_g, CV_32F);
    cv::GMat bg_b_32f = cv::gapi::convertTo(bg_b, CV_32F);
    cv::GMat bg_a_32f = cv::gapi::convertTo(bg_a, CV_32F);
    cv::GMat img_alpha_32f_inv = cv::gapi::subRC(1, img_alpha_32f);
    cv::GMat img_blend_r_32f = cv::gapi::mul(fg_r_32f, img_alpha_32f) + cv::gapi::mul(bg_r_32f, img_alpha_32f_inv);
    cv::GMat img_blend_g_32f = cv::gapi::mul(fg_g_32f, img_alpha_32f) + cv::gapi::mul(bg_g_32f, img_alpha_32f_inv);
    cv::GMat img_blend_b_32f = cv::gapi::mul(fg_b_32f, img_alpha_32f) + cv::gapi::mul(bg_b_32f, img_alpha_32f_inv);
    cv::GMat img_blend_a_32f = cv::gapi::mul(fg_a_32f, img_alpha_32f) + cv::gapi::mul(bg_a_32f, img_alpha_32f_inv);
    cv::GMat img_blend_r = cv::gapi::convertTo(img_blend_r_32f, CV_8U);
    cv::GMat img_blend_g = cv::gapi::convertTo(img_blend_g_32f, CV_8U);
    cv::GMat img_blend_b = cv::gapi::convertTo(img_blend_b_32f, CV_8U);
    cv::GMat img_blend_a = cv::gapi::convertTo(img_blend_a_32f, CV_8U);

    // blend CV type.
    int blend_depth = img_blend.depth();
    float blend_max = calc_value_max(img_blend);
    cv::GMat img_blend_r_conv = cv::gapi::convertTo(img_blend_r, blend_depth, blend_max / 255);
    cv::GMat img_blend_g_conv = cv::gapi::convertTo(img_blend_g, blend_depth, blend_max / 255);
    cv::GMat img_blend_b_conv = cv::gapi::convertTo(img_blend_b, blend_depth, blend_max / 255);
    cv::GMat img_blend_a_conv = cv::gapi::convertTo(img_blend_a, blend_depth, blend_max / 255);

    // merge.
    cv::GMat out_blend_1ch = img_blend_g_conv;
    cv::GMat out_blend_3ch = cv::gapi::merge3(img_blend_r_conv, img_blend_g_conv, img_blend_b_conv);
    cv::GMat out_a = cv::gapi::mulC(blend_max, cv::gapi::addC(1, (img_blend_g_conv ^ img_blend_g_conv)));
    cv::GMat out_blend_4ch = cv::gapi::merge4(img_blend_r_conv, img_blend_g_conv, img_blend_b_conv, out_a);

    cv::GComputation f_blend(cv::GIn(in_fg, in_bg, in_a), cv::GOut(out_blend_1ch, out_blend_3ch, out_blend_4ch));

    // auto pkg = cv::gapi::combine(
    //     cv::gapi::core::ocl::kernels(),
    //     cv::gapi::imgproc::ocl::kernels());
    auto pkg = cv::gapi::combine(
        cv::gapi::core::gpu::kernels(),
        cv::gapi::imgproc::gpu::kernels());

    // Go! G-API.
    cv::Mat img_blend_1ch;
    cv::Mat img_blend_3ch;
    cv::Mat img_blend_4ch;
    f_blend.apply(cv::gin(img_fg, img_bg, img_alpha), cv::gout(img_blend_1ch, img_blend_3ch, img_blend_4ch), cv::compile_args(pkg));

    // blend channels.
    int blend_ch = img_blend.channels();
    if (blend_ch == 1) {
        img_blend = img_blend_1ch;
    } else if (blend_ch == 4) {
        img_blend = img_blend_4ch;
    } else {
        img_blend = img_blend_3ch;
    }
}
#endif

#ifdef CUDA_ENABLE
auto make_cvmat_3ch_cuda = [](const cv::cuda::GpuMat &img) -> cv::cuda::GpuMat {
    cv::cuda::GpuMat img_3ch(img.size(), CV_MAKETYPE(img.depth(), 3));

    int ch = img.channels();
    switch (ch) {
    case 1:
        cv::cuda::merge(std::vector<cv::cuda::GpuMat>{ img, img, img }, img_3ch);
        break;
    case 2:
        {
            std::vector<cv::cuda::GpuMat> planes;
            cv::cuda::split(img, planes);
            cv::cuda::merge(std::vector<cv::cuda::GpuMat>{ planes[0], planes[0], planes[0] }, img_3ch);
        }
        break;
    case 3:
        img_3ch = img;
        break;
    case 4:
        {
            std::vector<cv::cuda::GpuMat> planes;
            cv::cuda::split(img, planes);
            cv::cuda::merge(std::vector<cv::cuda::GpuMat>{ planes[0], planes[1], planes[2] }, img_3ch);
        }
        break;
    default:
        break;
    }

    return img_3ch;
};

cv::cuda::GpuMat blend_image_3ch(const cv::cuda::GpuMat &img_fg_3ch, const cv::cuda::GpuMat &img_bg_3ch, const cv::cuda::GpuMat &img_alpha)
{
    if (img_fg_3ch.depth() != CV_32F || img_fg_3ch.channels() != 3
        || img_bg_3ch.depth() != CV_32F || img_bg_3ch.channels() != 3
        || img_alpha.depth() != CV_32F || img_alpha.channels() != 1
        || img_fg_3ch.size() != img_bg_3ch.size()
        || img_fg_3ch.size() != img_alpha.size()
    ) {
        return cv::cuda::GpuMat{};
    }

    // to 3ch.
    // 3.8 ms.
    cv::cuda::GpuMat a;
    cv::cuda::merge(std::vector<cv::cuda::GpuMat>{ img_alpha, img_alpha, img_alpha }, a);
    cv::cuda::GpuMat a_ones(img_alpha.size(), CV_MAKETYPE(img_alpha.depth(), img_alpha.channels()), cv::Scalar(1));
    cv::cuda::GpuMat img_alpha_inv;
    cv::cuda::subtract(a_ones, img_alpha, img_alpha_inv);
    cv::cuda::GpuMat a_inv;
    cv::cuda::merge(std::vector<cv::cuda::GpuMat>{ img_alpha_inv, img_alpha_inv, img_alpha_inv }, a_inv);

    // blend.
    // 6 ms.
    cv::cuda::GpuMat img_blend_3ch;
    cv::cuda::GpuMat img_blend_3ch_a;
    cv::cuda::multiply(img_fg_3ch, a, img_blend_3ch_a);
    cv::cuda::GpuMat img_blend_3ch_a_inv;
    cv::cuda::multiply(img_bg_3ch, a_inv, img_blend_3ch_a_inv);
    cv::cuda::add(img_blend_3ch_a, img_blend_3ch_a_inv, img_blend_3ch);

    return img_blend_3ch;
}

void blend_image(const cv::cuda::GpuMat &img_fg, const cv::cuda::GpuMat &img_bg, const cv::cuda::GpuMat &img_alpha, cv::cuda::GpuMat &img_blend)
{
    // static double dt_ns_sum = 0.0f;
    // static double sum_count = 0.0f;
    // auto st = std::chrono::steady_clock::now();

    // resize.
    cv::Size blend_size(img_blend.size().width, img_blend.size().height);
    cv::cuda::GpuMat img_fg_resized;
    cv::cuda::GpuMat img_bg_resized;
    cv::cuda::GpuMat img_alpha_resized;
    auto inter_type = [](const cv::cuda::GpuMat &in_img, const cv::cuda::GpuMat &out_img) -> cv::InterpolationFlags {
        return ((in_img.size().width < out_img.size().width) && (in_img.size().height < out_img.size().height)
            ? cv::INTER_NEAREST
            : cv::INTER_AREA);
    };
    cv::cuda::resize(img_fg, img_fg_resized, blend_size, 0.0, 0.0, inter_type(img_fg, img_blend));
    cv::cuda::resize(img_bg, img_bg_resized, blend_size, 0.0, 0.0, inter_type(img_bg, img_blend));
    cv::cuda::resize(img_alpha, img_alpha_resized, blend_size, 0.0, 0.0, inter_type(img_alpha, img_blend));

    // 3ch.
    cv::cuda::GpuMat img_fg_3ch = make_cvmat_3ch_cuda(img_fg_resized);
    cv::cuda::GpuMat img_bg_3ch = make_cvmat_3ch_cuda(img_bg_resized);

    // CV_32F.
    cv::cuda::GpuMat img_fg_32fc3;
    cv::cuda::GpuMat img_bg_32fc3;
    cv::cuda::GpuMat img_alpha_32fc1;
    img_fg_3ch.convertTo(img_fg_32fc3, CV_32F, 1 / calc_value_max(img_fg));
    img_bg_3ch.convertTo(img_bg_32fc3, CV_32F, 1 / calc_value_max(img_bg));
    img_alpha_resized.convertTo(img_alpha_32fc1, CV_32F, 1 / calc_value_max(img_alpha));

    // blend.
    cv::cuda::GpuMat img_blend_32fc3 = blend_image_3ch(img_fg_32fc3, img_bg_32fc3, img_alpha_32fc1);

    // blend channels.
    cv::cuda::GpuMat a_ones = cv::cuda::GpuMat(blend_size, CV_32FC1, cv::Scalar(1));
    cv::cuda::GpuMat img_blend_tmp;
    std::vector<cv::cuda::GpuMat> planes_blend;
    int blend_ch = img_blend.channels();
    if (blend_ch == 1) {
        cv::cuda::cvtColor(img_blend_32fc3, img_blend_tmp, cv::COLOR_RGB2GRAY, blend_ch);
    } else if (blend_ch == 2) {
        cv::cuda::cvtColor(img_blend_32fc3, img_blend_tmp, cv::COLOR_RGB2GRAY, 1);
        cv::cuda::merge(std::vector<cv::cuda::GpuMat>{ img_blend_tmp, a_ones }, img_blend_tmp);
    } else if (blend_ch == 3) {
        img_blend_tmp = img_blend_32fc3;
    } else if (blend_ch == 4) {
        cv::cuda::split(img_blend_32fc3, planes_blend);
        cv::cuda::merge(std::vector<cv::cuda::GpuMat>{ planes_blend[0], planes_blend[1], planes_blend[2], a_ones }, img_blend_tmp);
    } else {
        img_blend_tmp = img_blend_32fc3;
    }

    // blend CV type.
    int blend_depth = img_blend.depth();
    int cv_type = CV_MAKETYPE(blend_depth, blend_ch);
    float blend_max = calc_value_max(img_blend);
    img_blend_tmp.convertTo(img_blend, cv_type, blend_max);

    // auto et = std::chrono::steady_clock::now();
    // auto dt = et - st;
    // auto dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count();
    // dt_ns_sum += dt_ns;
    // sum_count++;
    // std::cout << "blend_image(): " << (dt_ns_sum / sum_count) / 1'000'000.0f << " (msec)" << std::endl;
}
#endif

auto make_img_margin = [](cv::Mat src, auto ci, auto mi, auto init_val) -> cv::Mat {
    auto mat_type = CV_MAKETYPE(src.depth(), src.channels());
    cv::Mat dst = cv::Mat{ cv::Size(mi.outer_width, mi.outer_height), mat_type, init_val };

    cv::Mat img_crop(src, cv::Rect(ci.pos_x, ci.pos_y, ci.inner_width, ci.inner_height));
    cv::Mat img_inner;
    cv::resize(img_crop, img_inner, cv::Size(mi.inner_width, mi.inner_height), 0.0, 0.0, cv::INTER_AREA);
    cv::Mat img_roi(dst, cv::Rect(mi.pos_x, mi.pos_y, mi.inner_width, mi.inner_height));
    img_inner.copyTo(img_roi);

    return dst;
};

auto gen_img_edge = [](cv::Mat src) -> cv::Mat {
    cv::Mat dst;

    // -> Gary Scale.
    cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);

    // detect edge.
    {
        constexpr auto canny_th1 = 100;
        constexpr auto canny_th2 = 200;
        cv::Canny(dst, dst, canny_th1, canny_th2);
    }

    // morph.
    {
        constexpr auto k_size = 1;
        cv::Mat d_kernel = getStructuringElement(
            cv::MORPH_RECT, // cv::MORPH_RECT, cv::MORPH_CROSS, cv::MORPH_ELLIPSE.
            cv::Size( 2 * k_size + 1, 2 * k_size + 1 ),
            cv::Point( k_size, k_size )
        );
        cv::dilate(dst, dst, d_kernel);
    }

    // binarization.
    {
        constexpr auto th = 128;
        constexpr auto th_max = 255;
        cv::threshold(dst, dst, th, th_max, cv::THRESH_BINARY);
    }

    return dst;
};

#ifdef USE_COMPLETION_LIB
struct st_zup_image_stack
{
	std::unique_ptr<uint8_t []> guide_stack;
	std::unique_ptr<uint8_t []> src_stack;
	std::unique_ptr<uint8_t []> global_mask_img_stack;
	std::unique_ptr<uint8_t []> dst_stack;
};

cv::Mat zup_core(const cv::Mat &src, const cv::Mat &guide, st_zup_image_stack &img_stack, int32_t frame_num)
{
    assert(src.size() == guide.size());

#ifdef CUDA_ENABLE
    bool gpu_enable = true;
#else
    bool gpu_enable = false;
#endif

    bool is_16bit = (src.depth() == CV_16U || src.depth() == CV_16S);
    cv::Size img_size = src.size();
	cv::Mat global_mask_img(img_size, CV_8U, cv::Scalar(1));	// 全ピクセルが有効なglobal_mask_img.

    // WMF.
#if 0
    cv::Mat img_wmf(img_size, src.type());

	// WMFのパラメータの設定.
	int32_t wmf_ksize = 3;
	float wmf_sigma = 15.0f;
	int32_t wmf_guide_ch = 1;
	uint8_t wmf_weight_coef[1] = { 64 };
	uint8_t wmf_local_mask[3 * 3] = { 1,1,1,1,1,1,1,1,1 };

    if (is_16bit) {
        complib_unit_twmf_u16(guide.data, (uint16_t *)src.data, global_mask_img.data, (uint16_t *)img_wmf.data,
            nullptr, nullptr, nullptr, 1, 1,
            img_size.width, img_size.height, wmf_guide_ch, wmf_ksize, wmf_sigma, wmf_weight_coef, reinterpret_cast<uint8_t*>(wmf_local_mask), false, gpu_enable);
    } else {
        complib_unit_twmf_u8(guide.data, src.data, global_mask_img.data, img_wmf.data,
            nullptr, nullptr, nullptr, 1, 1,
            img_size.width, img_size.height, wmf_guide_ch, wmf_ksize, wmf_sigma, wmf_weight_coef, reinterpret_cast<uint8_t*>(wmf_local_mask), false, gpu_enable);
    }

    cv::Mat dst = img_wmf;

    // TWMF.
#else
    cv::Mat img_twmf(img_size, src.type());

	// TWMFのパラメータの設定.
	const int32_t twmf_ksize = 3;
	float twmf_sigma = 15.0f;
	int32_t twmf_guide_ch = 1;
	// uint8_t twmf_weight_coef[1] = { 64 };
	// uint8_t twmf_local_mask[twmf_ksize * twmf_ksize] = { 1,1,1,1,1,1,1,1,1 };

	const int32_t stack_num = 2;
    const int32_t pre_stack_num = stack_num - 1;
	uint8_t twmf_stack_weight_coef[stack_num] = { 64 , 64 };
	uint8_t twmf_stack_local_mask[stack_num][twmf_ksize * twmf_ksize] = {
		{1,1,1,1,1,1,1,1,1},
		{1,1,1,1,1,1,1,1,1}
	};
	uint32_t guide_num = img_size.width * img_size.height * twmf_guide_ch;
	uint32_t guide_nbytes = guide_num * (is_16bit ? sizeof(uint16_t) : sizeof(uint8_t));
	uint32_t src_num = img_size.width * img_size.height * twmf_guide_ch;
	uint32_t src_nbytes = src_num * (is_16bit ? sizeof(uint16_t) : sizeof(uint8_t));
	uint32_t global_mask_img_num    = img_size.width * img_size.height * twmf_guide_ch;
	uint32_t global_mask_img_nbytes = global_mask_img_num * (is_16bit ? sizeof(uint16_t) : sizeof(uint8_t));

	auto &guide_stack = img_stack.guide_stack;
	auto &src_stack = img_stack.src_stack;
	auto &global_mask_img_stack = img_stack.global_mask_img_stack;
	if (!guide_stack) guide_stack = std::make_unique<uint8_t []>(guide_nbytes * pre_stack_num);
	if (!src_stack) src_stack = std::make_unique<uint8_t []>(src_nbytes * pre_stack_num);
	if (!global_mask_img_stack) global_mask_img_stack = std::make_unique<uint8_t []>(global_mask_img_nbytes * pre_stack_num);

    int32_t now_stack_num = std::min(frame_num + 1, stack_num);
    bool iir_enable = true;
    if (is_16bit) {
        complib_unit_twmf_u16(guide.data, reinterpret_cast<uint16_t*>(src.data), global_mask_img.data, reinterpret_cast<uint16_t*>(img_twmf.data),
            guide_stack.get(), reinterpret_cast<uint16_t*>(src_stack.get()), global_mask_img_stack.get(), now_stack_num, stack_num,
            img_size.width, img_size.height, twmf_guide_ch, twmf_ksize, twmf_sigma, twmf_stack_weight_coef, reinterpret_cast<uint8_t*>(twmf_stack_local_mask), iir_enable, gpu_enable);
    } else {
        complib_unit_twmf_u8(guide.data, src.data, global_mask_img.data, img_twmf.data,
            guide_stack.get(), src_stack.get(), global_mask_img_stack.get(), now_stack_num, stack_num,
            img_size.width, img_size.height, twmf_guide_ch, twmf_ksize, twmf_sigma, twmf_stack_weight_coef, reinterpret_cast<uint8_t*>(twmf_stack_local_mask), iir_enable, gpu_enable);
    }

    cv::Mat dst = img_twmf;
#endif

    return dst;
}
cv::Mat zup_iter(const cv::Mat &src, const cv::Mat &guide, std::vector<st_zup_image_stack> &img_stacks, int32_t frame_num, int level)
{
    auto src_pixels = src.size().width * src.size().height;
    auto guide_pixels = guide.size().width * guide.size().height;

    if (img_stacks.size() < level) {
        img_stacks.push_back({});
    }

    auto zup = [](const cv::Mat &src, const cv::Mat &guide, st_zup_image_stack &img_stack, int32_t frame_num) -> cv::Mat {
        cv::Mat src_scaled;
        cv::resize(src, src_scaled, guide.size(), 0.0, 0.0, cv::INTER_LINEAR);
        cv::Mat dst = zup_core(src_scaled, guide, img_stack, frame_num);
        return dst;
    };

    cv::Mat dst;
    if (src_pixels <= guide_pixels / 4) {
        cv::Mat guide_half;
        cv::resize(guide, guide_half, guide.size() / 2, 0.0, 0.0, cv::INTER_LINEAR);
        cv::Mat dst_half = zup_iter(src, guide_half, img_stacks, frame_num, level + 1);
        dst = zup(dst_half, guide, img_stacks[level - 1], frame_num);
    } else {
        dst = zup(src, guide, img_stacks[level - 1], frame_num);
    }

    return dst;
}
cv::Mat zup(const cv::Mat &src, const cv::Mat &guide, bool hfbs_enable, bool reset_stacks)
#if 1
{
#ifndef NDEBUG
    static StopWatch sw;
    sw.start();
#endif

#ifdef CUDA_ENABLE
    bool gpu_enable = true;
#else
    bool gpu_enable = false;
#endif

    auto src_depth = src.depth();
    bool is_16bit = (src_depth == CV_16U || src_depth == CV_16S);

    static st_zup_image_stack img_stacks;   // 暫定. 本当は繰り返し処理の外側か、クラスのメンバ変数で宣言する.
    static int32_t frame_num = 0;           // 暫定. 本当は繰り返し処理の外側か、クラスのメンバ変数で宣言する.

	const int32_t stack_num = 2;
    const int32_t pre_stack_num = stack_num - 1;
    int32_t now_stack_num = std::min(frame_num + 1, stack_num);
    bool iir_enable = true;

    cv::Mat img_zup(guide.size(), src.type());
    cv::Mat global_mask_img(src.size(), CV_8U, cv::Scalar(1));	// 全ピクセルが有効なglobal_mask_img.

    if (reset_stacks) {
        uint32_t guide_num = guide.size().width * guide.size().height * guide.channels();
        uint32_t guide_nbytes = guide_num * guide.elemSize1();
        uint32_t src_num = src.size().width * src.size().height * src.channels();
        uint32_t src_nbytes = src_num * src.elemSize1();
        uint32_t global_mask_img_num    = global_mask_img.size().width * global_mask_img.size().height * global_mask_img.channels();
        uint32_t global_mask_img_nbytes = global_mask_img_num * global_mask_img.elemSize1();
        uint32_t dst_num    = img_zup.size().width * img_zup.size().height * img_zup.channels();
        uint32_t dst_nbytes = dst_num * img_zup.elemSize1();

        img_stacks.guide_stack = std::make_unique<uint8_t []>(guide_nbytes * pre_stack_num);
        img_stacks.src_stack = std::make_unique<uint8_t []>(src_nbytes * pre_stack_num);
        img_stacks.global_mask_img_stack = std::make_unique<uint8_t []>(global_mask_img_nbytes * pre_stack_num);
        img_stacks.dst_stack = std::make_unique<uint8_t []>(dst_nbytes * pre_stack_num);
    }
    if (!img_stacks.dst_stack) {
        return img_zup;
    }

    auto guide_prestack_img = img_stacks.guide_stack.get();
    auto src_prestack_img = img_stacks.src_stack.get();
    auto global_mask_prestack_img = img_stacks.global_mask_img_stack.get();
    auto dst_prestack_img = img_stacks.dst_stack.get();

    if (hfbs_enable) {
		if (is_16bit) {
			complib_pipeline_compedge_u16(
				guide.data, reinterpret_cast<uint16_t*>(src.data), global_mask_img.data, reinterpret_cast<uint16_t*>(img_zup.data),
				guide_prestack_img, reinterpret_cast<uint16_t*>(src_prestack_img), global_mask_prestack_img, reinterpret_cast<uint16_t*>(dst_prestack_img), now_stack_num, stack_num,
				guide.channels(), guide.size().width, guide.size().height, src.size().width, src.size().height, gpu_enable);
		}
		else {
			complib_pipeline_compedge_u8(
				guide.data, src.data, global_mask_img.data, img_zup.data,
				guide_prestack_img, src_prestack_img, global_mask_prestack_img, dst_prestack_img, now_stack_num, stack_num,
				guide.channels(), guide.size().width, guide.size().height, src.size().width, src.size().height, gpu_enable);
		}
    } else {
		if (is_16bit) {
			complib_pipeline_compabs_u16(
				guide.data, reinterpret_cast<uint16_t*>(src.data), global_mask_img.data, reinterpret_cast<uint16_t*>(img_zup.data),
				guide_prestack_img, reinterpret_cast<uint16_t*>(src_prestack_img), global_mask_prestack_img, reinterpret_cast<uint16_t*>(dst_prestack_img), now_stack_num, stack_num,
				guide.channels(), guide.size().width, guide.size().height, src.size().width, src.size().height, gpu_enable);
		}
		else {
			complib_pipeline_compabs_u8(
				guide.data, src.data, global_mask_img.data, img_zup.data,
				guide_prestack_img, src_prestack_img, global_mask_prestack_img, dst_prestack_img, now_stack_num, stack_num,
				guide.channels(), guide.size().width, guide.size().height, src.size().width, src.size().height, gpu_enable);
		}
    }
    frame_num++;

    cv::Mat dst = img_zup;

#ifndef NDEBUG
    auto [lap_cur, lap_ave] = sw.lap();
    std::cout << "ZUP: " << lap_cur << ", " << lap_ave << "(msec)" << std::endl;
#endif

    return dst;
}
#elif 0
{
#ifndef NDEBUG
    static StopWatch sw;
    sw.start();
#endif

    static std::vector<st_zup_image_stack> img_stacks;  // 暫定. 本当は繰り返し処理の外側か、クラスのメンバ変数で宣言する.
    static int32_t frame_num = 0;                       // 暫定. 本当は繰り返し処理の外側か、クラスのメンバ変数で宣言する.
    if (reset_stacks) img_stacks.clear();
    cv::Mat img_zup_iter = zup_iter(src, guide, img_stacks, frame_num, 1);
    frame_num++;

    // HFBSのパラメータの設定.
    uint32_t hfbs_sigma_spatial = 8;
    uint32_t hfbs_sigma_luma = 4;
    float hfbs_lambda = 128.0f;
    uint32_t hfbs_optim_iter_num = 256;

#ifdef CUDA_ENABLE
    bool gpu_enable = true;
#else
    bool gpu_enable = false;
#endif

    bool is_16bit = (src.depth() == CV_16U || src.depth() == CV_16S);
    cv::Size img_size = guide.size();

    cv::Mat img_conf(img_size, CV_8UC1, cv::Scalar(255));
    cv::Mat img_gray;
    cv::cvtColor(guide, img_gray, cv::COLOR_RGB2GRAY);
    cv::Mat img_hfbs;
    img_hfbs.create(img_size, src.type());
    if (is_16bit) {
        complib_unit_thfbs_u16(img_gray.data, (uint16_t *)img_zup_iter.data, img_conf.data, (uint16_t *)img_hfbs.data,
			nullptr, nullptr, nullptr, nullptr, 1, 1,
            img_size.width, img_size.height, hfbs_sigma_spatial, hfbs_sigma_luma, hfbs_lambda,
			hfbs_optim_iter_num, 0, 0, 1, gpu_enable, false);
    } else {
        complib_unit_thfbs_u8(img_gray.data, img_zup_iter.data, img_conf.data, img_hfbs.data,
			nullptr, nullptr, nullptr, nullptr, 1, 1,
			img_size.width, img_size.height, hfbs_sigma_spatial, hfbs_sigma_luma, hfbs_lambda,
			hfbs_optim_iter_num, 0, 0, 1, gpu_enable, false);
    }

    cv::Mat dst;
    if (hfbs_enable) {
        dst = img_hfbs;
    } else {
        dst = img_zup_iter;
    }

#ifndef NDEBUG
    auto [lap_cur, lap_ave] = sw.lap();
    std::cout << "ZUP: " << lap_cur << ", " << lap_ave << "(msec)" << std::endl;
#endif

    return dst;
}
#else
{
    static StopWatch sw;
    sw.start();

    // HFBSのパラメータの設定.
    uint32_t hfbs_sigma_spatial = 8;
    uint32_t hfbs_sigma_luma = 4;
    float hfbs_lambda = 128.0f;
    uint32_t hfbs_optim_iter_num = 256;

    if (src.depth() == CV_16U || src.depth() == CV_16S) src.convertTo(src, CV_8U, 1.0 / 256.0);
    cv::Size img_size = guide.size();
    cv::resize(src, src, img_size);
    cv::Mat dst;
    // cv::ximgproc::fastBilateralSolverFilter(guide, src, cv::Mat(img_size, CV_32FC1, cv::Scalar(1)), dst, 8.0, 8.0, 8.0, 128.0, 25);
    if (true) {
        cv::Mat img_conf(img_size, CV_8UC1, cv::Scalar(255));
        cv::Mat img_gray;
        cv::cvtColor(guide, img_gray, cv::COLOR_RGB2GRAY);
        std::vector<cv::Mat> planes;
        cv::split(src, planes);
        std::vector<cv::Mat> img_ary;
        for (auto i = 0; i < planes.size(); i++) {
            cv::Mat img_src = planes[i];
            // cv::imwrite("guide.png", img_gray);
            // cv::imwrite("img_conf.png", img_conf);
            // std::string str = "src" + std::to_string(i) + ".png";
            // cv::imwrite(str.c_str(), img_src);

            // HFBS_GPU_uint8.
            cv::Mat img_hfbs;
            img_hfbs.create(img_size, img_src.type());
#ifdef CUDA_ENABLE
            bool gpu_enable = true;
#else
            bool gpu_enable = false;
#endif
            complib_unit_thfbs_u8(img_gray.data, img_src.data, img_conf.data, img_hfbs.data,
				nullptr, nullptr, nullptr, nullptr, 1, 1,
                img_size.width, img_size.height, hfbs_sigma_spatial, hfbs_sigma_luma, hfbs_lambda,
				hfbs_optim_iter_num, 0,0,1,gpu_enable, false);

            img_ary.emplace_back(img_hfbs);
            // str = "dst" + std::to_string(i) + ".png";
            // cv::imwrite(str.c_str(), img_ary[i]);
        }
        cv::merge(img_ary, dst);
    }

    auto [lap_cur, lap_ave] = sw.lap();
    std::cout << "HFBS: " << lap_cur << ", " << lap_ave << "(msec)" << std::endl;

    return dst;
}
#endif
#endif

}
