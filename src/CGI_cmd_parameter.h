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

namespace CGICmd
{

// Shutter.
using shutter_list = std::list<std::pair<int, std::string>>;
extern shutter_list exposure_exposure_time_5994p;
extern shutter_list exposure_exposure_time_5000p;
extern shutter_list exposure_exposure_time_2997p;
extern shutter_list exposure_exposure_time_2500p;
extern shutter_list exposure_exposure_time_2400p;
extern shutter_list exposure_exposure_time_2398p;
extern std::unordered_map<std::string, shutter_list> exposure_exposure_time;

using angle_list = shutter_list;
extern angle_list exposure_angle;

// White Balance.
using color_temp_list = shutter_list;
extern color_temp_list white_balance_color_temp;

// ISO.
using iso_list = shutter_list;
extern iso_list exposure_iso;

using gain_list = shutter_list;
extern gain_list exposure_gain;

using EI_list = shutter_list;
extern EI_list exposure_exposure_index_iso800;
extern EI_list exposure_exposure_index_iso12800;
extern std::unordered_map<std::string, EI_list> exposure_exposure_index;

// IRIS.
constexpr auto iris_EV_div = 3.0f;
const auto iris_sqrt2 = std::sqrt(2.0f);
constexpr auto iris_unit_EV_inv = 256.0f;
constexpr auto iris_F1_value = 32768.0f;
const auto iris_k_multi = std::pow(iris_sqrt2, 1.0f / iris_EV_div);
const auto iris_k_add = -(iris_unit_EV_inv / iris_EV_div);
inline auto calc_fnum = [](auto val) -> auto {
    auto fnum = std::pow(CGICmd::iris_k_multi, (val - CGICmd::iris_F1_value) / CGICmd::iris_k_add);
    return fnum;
};

// ND.
using nd_list_t = shutter_list;
extern nd_list_t exposure_nd;

}   // namespace CGICmd
