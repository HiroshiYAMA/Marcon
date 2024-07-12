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

#include "CGI.h"

namespace CGICmd
{

inline auto conv_str2num = [](const std::string &str, int &num) -> bool {
    try
    {
        num = std::stoi(str);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }

    return true;
};

inline auto conv_json_str2num = [](const njson& j, const char *key, int &num) -> bool {
    std::string str;
    json_get_val(j, key, str);
    auto ret = conv_str2num(str, num);

    return ret;
};

inline auto conv_range2str = [](const CGICmd::st_Range &range) -> std::string {
    std::stringstream ss;
    ss << range.min << ", " << range.max;
    std::string range_str = ss.str();

    return range_str;
};

inline auto conv_json_str2range = [](const njson& j, const char *key, CGICmd::st_Range &range) -> bool {
    std::string range_str;
    json_get_val(j, key, range_str);
    std::stringstream ss{range_str};
    std::string str;
    std::vector<std::string> v;
    while (std::getline(ss, str, ',')) v.push_back(str);

    bool ret = false;
    if (v.size() >= 2) {
        ret = conv_str2num(v[0], range.min) && conv_str2num(v[1], range.max);
    }

    return ret;
};

// json <---> st_Imaging.
constexpr auto EXPOSURE_ANGLE = "ExposureAngle";
constexpr auto EXPOSURE_ANGLE_RANGE = "ExposureAngleRange";
constexpr auto EXPOSURE_ECS = "ExposureECS";
constexpr auto EXPOSURE_ECS_VALUE = "ExposureECSValue";
constexpr auto EXPOSURE_EXPOSURE_TIME = "ExposureExposureTime";
constexpr auto EXPOSURE_EXPOSURE_TIME_RANGE = "ExposureExposureTimeRange";
constexpr auto EXPOSURE_SHUTTER_MODE_STATE = "ExposureShutterModeState";
void to_json(njson& j, const CGICmd::st_Imaging& p) {
    j[EXPOSURE_ANGLE] = std::to_string(p.ExposureAngle);
    j[EXPOSURE_ANGLE_RANGE] = conv_range2str(p.ExposureAngleRange);
    j[EXPOSURE_ECS] = std::to_string(p.ExposureECS);
    j[EXPOSURE_ECS_VALUE] = std::to_string(p.ExposureECSValue);
    j[EXPOSURE_EXPOSURE_TIME] = std::to_string(p.ExposureExposureTime);
    j[EXPOSURE_EXPOSURE_TIME_RANGE] = conv_range2str(p.ExposureExposureTimeRange);
    j[EXPOSURE_SHUTTER_MODE_STATE] = p.ExposureShutterModeState;
}
void from_json(const njson& j, CGICmd::st_Imaging& p) {
    conv_json_str2num(j, EXPOSURE_ANGLE, p.ExposureAngle);
    conv_json_str2range(j, EXPOSURE_ANGLE_RANGE, p.ExposureAngleRange);
    conv_json_str2num(j, EXPOSURE_ECS, p.ExposureECS);
    conv_json_str2num(j, EXPOSURE_ECS_VALUE, p.ExposureECSValue);
    conv_json_str2num(j, EXPOSURE_EXPOSURE_TIME, p.ExposureExposureTime);
    conv_json_str2range(j, EXPOSURE_EXPOSURE_TIME_RANGE, p.ExposureExposureTimeRange);
    json_get_val(j, EXPOSURE_SHUTTER_MODE_STATE, p.ExposureShutterModeState);
}

}   // namespace CGICmd.
