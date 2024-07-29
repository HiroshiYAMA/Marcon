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
// shutter.
constexpr auto EXPOSURE_ANGLE = "ExposureAngle";
constexpr auto EXPOSURE_ANGLE_RANGE = "ExposureAngleRange";
constexpr auto EXPOSURE_ECS = "ExposureECS";
constexpr auto EXPOSURE_ECS_RANGE = "ExposureECSRange";
constexpr auto EXPOSURE_ECS_VALUE = "ExposureECSValue";
constexpr auto EXPOSURE_EXPOSURE_TIME = "ExposureExposureTime";
constexpr auto EXPOSURE_EXPOSURE_TIME_RANGE = "ExposureExposureTimeRange";
constexpr auto EXPOSURE_SHUTTER_MODE_STATE = "ExposureShutterModeState";
// white balance.
constexpr auto WHITE_BALANCE_COLOR_TEMP = "WhiteBalanceColorTemp";
constexpr auto WHITE_BALANCE_COLOR_TEMP_CURRENT = "WhiteBalanceColorTempCurrent";
constexpr auto WHITE_BALANCE_MODE = "WhiteBalanceMode";
constexpr auto WHITE_BALANCE_GAIN_TEMP = "WhiteBalanceGainTemp";
// ISO.
constexpr auto EXPOSURE_AGC_ENABLE = "ExposureAGCEnable";
constexpr auto EXPOSURE_BASE_ISO = "ExposureBaseISO";
constexpr auto EXPOSURE_BASE_ISO_PMT = "ExposureBaseISOPmt";
constexpr auto EXPOSURE_BASE_SENSITIVITY = "ExposureBaseSensitivity";
constexpr auto EXPOSURE_EXPOSURE_INDEX = "ExposureExposureIndex";
constexpr auto EXPOSURE_EXPOSURE_INDEX_PMT = "ExposureExposureIndexPmt";
constexpr auto EXPOSURE_GAIN = "ExposureGain";
constexpr auto EXPOSURE_GAIN_TEMPORARY = "ExposureGainTemporary";
constexpr auto EXPOSURE_ISO = "ExposureISO";
constexpr auto EXPOSURE_ISO_TEMPORARY = "ExposureISOTemporary";
constexpr auto EXPOSURE_ISO_GAIN_MODE = "ExposureISOGainMode";
// IRIS.
constexpr auto EXPOSURE_AUTO_IRIS = "ExposureAutoIris";
constexpr auto EXPOSURE_FNUMBER = "ExposureFNumber";
constexpr auto EXPOSURE_IRIS = "ExposureIris";
constexpr auto EXPOSURE_IRIS_RANGE = "ExposureIrisRange";
// ND.
constexpr auto EXPOSURE_AUTO_ND_FILTER_ENABLE = "ExposureAutoNDFilterEnable";
constexpr auto EXPOSURE_ND_CLEAR = "ExposureNDClear";
constexpr auto EXPOSURE_ND_VARIABLE = "ExposureNDVariable";
//
void to_json(njson& j, const CGICmd::st_Imaging& p) {
    // shutter.
    j[EXPOSURE_ANGLE] = std::to_string(p.ExposureAngle);
    j[EXPOSURE_ANGLE_RANGE] = conv_range2str(p.ExposureAngleRange);
    j[EXPOSURE_ECS] = std::to_string(p.ExposureECS);
    j[EXPOSURE_ECS_RANGE] = conv_range2str(p.ExposureECSRange);
    j[EXPOSURE_ECS_VALUE] = std::to_string(p.ExposureECSValue);
    j[EXPOSURE_EXPOSURE_TIME] = std::to_string(p.ExposureExposureTime);
    j[EXPOSURE_EXPOSURE_TIME_RANGE] = conv_range2str(p.ExposureExposureTimeRange);
    j[EXPOSURE_SHUTTER_MODE_STATE] = p.ExposureShutterModeState;
    // white balance.
    j[WHITE_BALANCE_COLOR_TEMP] = std::to_string(p.WhiteBalanceColorTemp);
    j[WHITE_BALANCE_COLOR_TEMP_CURRENT] = std::to_string(p.WhiteBalanceColorTempCurrent);
    j[WHITE_BALANCE_MODE] = p.WhiteBalanceMode;
    j[WHITE_BALANCE_GAIN_TEMP] = p.WhiteBalanceGainTemp;
    // ISO.
    j[EXPOSURE_AGC_ENABLE] = p.ExposureAGCEnable;
    j[EXPOSURE_BASE_ISO] = p.ExposureBaseISO;
    j[EXPOSURE_BASE_ISO_PMT] = p.ExposureBaseISOPmt;
    j[EXPOSURE_BASE_SENSITIVITY] = p.ExposureBaseSensitivity;
    j[EXPOSURE_EXPOSURE_INDEX] = std::to_string(p.ExposureExposureIndex);
    j[EXPOSURE_EXPOSURE_INDEX_PMT] = p.ExposureExposureIndexPmt;
    j[EXPOSURE_GAIN] = std::to_string(p.ExposureGain);
    j[EXPOSURE_GAIN_TEMPORARY] = std::to_string(p.ExposureGainTemporary);
    j[EXPOSURE_ISO] = std::to_string(p.ExposureISO);
    j[EXPOSURE_ISO_TEMPORARY] = std::to_string(p.ExposureISOTemporary);
    j[EXPOSURE_ISO_GAIN_MODE] = p.ExposureISOGainMode;
    // IRIS.
    j[EXPOSURE_AUTO_IRIS] = p.ExposureAutoIris;
    j[EXPOSURE_FNUMBER] = std::to_string(p.ExposureFNumber);
    j[EXPOSURE_IRIS] = std::to_string(p.ExposureIris);
    j[EXPOSURE_IRIS_RANGE] = conv_range2str(p.ExposureIrisRange);
    // ND.
    j[EXPOSURE_AUTO_ND_FILTER_ENABLE] = p.ExposureAutoNDFilterEnable;
    j[EXPOSURE_ND_CLEAR] = p.ExposureNDClear;
    j[EXPOSURE_ND_VARIABLE] = std::to_string(p.ExposureNDVariable);
}
void from_json(const njson& j, CGICmd::st_Imaging& p) {
    // shutter.
    conv_json_str2num(j, EXPOSURE_ANGLE, p.ExposureAngle);
    conv_json_str2range(j, EXPOSURE_ANGLE_RANGE, p.ExposureAngleRange);
    conv_json_str2num(j, EXPOSURE_ECS, p.ExposureECS);
    conv_json_str2range(j, EXPOSURE_ECS_RANGE, p.ExposureECSRange);
    conv_json_str2num(j, EXPOSURE_ECS_VALUE, p.ExposureECSValue);
    conv_json_str2num(j, EXPOSURE_EXPOSURE_TIME, p.ExposureExposureTime);
    conv_json_str2range(j, EXPOSURE_EXPOSURE_TIME_RANGE, p.ExposureExposureTimeRange);
    json_get_val(j, EXPOSURE_SHUTTER_MODE_STATE, p.ExposureShutterModeState);
    // white balance.
    conv_json_str2num(j, WHITE_BALANCE_COLOR_TEMP, p.WhiteBalanceColorTemp);
    conv_json_str2num(j, WHITE_BALANCE_COLOR_TEMP_CURRENT, p.WhiteBalanceColorTempCurrent);
    json_get_val(j, WHITE_BALANCE_MODE, p.WhiteBalanceMode);
    json_get_val(j, WHITE_BALANCE_GAIN_TEMP, p.WhiteBalanceGainTemp);
    // ISO.
    json_get_val(j, EXPOSURE_AGC_ENABLE, p.ExposureAGCEnable);
    json_get_val(j, EXPOSURE_BASE_ISO, p.ExposureBaseISO);
    json_get_val(j, EXPOSURE_BASE_ISO_PMT, p.ExposureBaseISOPmt);
    json_get_val(j, EXPOSURE_BASE_SENSITIVITY, p.ExposureBaseSensitivity);
    conv_json_str2num(j, EXPOSURE_EXPOSURE_INDEX, p.ExposureExposureIndex);
    json_get_val(j, EXPOSURE_EXPOSURE_INDEX_PMT, p.ExposureExposureIndexPmt);
    conv_json_str2num(j, EXPOSURE_GAIN, p.ExposureGain);
    conv_json_str2num(j, EXPOSURE_GAIN_TEMPORARY, p.ExposureGainTemporary);
    conv_json_str2num(j, EXPOSURE_ISO, p.ExposureISO);
    conv_json_str2num(j, EXPOSURE_ISO_TEMPORARY, p.ExposureISOTemporary);
    json_get_val(j, EXPOSURE_ISO_GAIN_MODE, p.ExposureISOGainMode);
    // IRIS.
    json_get_val(j, EXPOSURE_AUTO_IRIS, p.ExposureAutoIris);
    conv_json_str2num(j, EXPOSURE_FNUMBER, p.ExposureFNumber);
    conv_json_str2num(j, EXPOSURE_IRIS, p.ExposureIris);
    conv_json_str2range(j, EXPOSURE_IRIS_RANGE, p.ExposureIrisRange);
    // ND.
    json_get_val(j, EXPOSURE_AUTO_ND_FILTER_ENABLE, p.ExposureAutoNDFilterEnable);
    json_get_val(j, EXPOSURE_ND_CLEAR, p.ExposureNDClear);
    conv_json_str2num(j, EXPOSURE_ND_VARIABLE, p.ExposureNDVariable);
}



// json <---> st_Network.
constexpr auto HTTP_PORT = "HttpPort";
constexpr auto HOST_NAME = "Hostname";
constexpr auto MAC_ADDRESS = "MacAddress";
constexpr auto CAMERA_NAME = "CameraName";
//
void to_json(njson& j, const CGICmd::st_Network& p) {
    j[HTTP_PORT] = std::to_string(p.HttpPort);
    j[HOST_NAME] = p.Hostname;
    j[MAC_ADDRESS] = p.MacAddress;
    j[CAMERA_NAME] = p.CameraName;
}
void from_json(const njson& j, CGICmd::st_Network& p) {
    conv_json_str2num(j, HTTP_PORT, p.HttpPort);
    json_get_val(j, HOST_NAME, p.Hostname);
    json_get_val(j, MAC_ADDRESS, p.MacAddress);
    json_get_val(j, CAMERA_NAME, p.CameraName);
}



// json <---> st_Srt.
constexpr auto SRT_ENCRYPTION = "SrtEncryption";
constexpr auto SRT_PASSPHRASE = "SrtPassphrase";
constexpr auto SRT_PASSPHRASE_USED = "SrtPassphraseUsed";
//
constexpr auto SRT_LISTEN_PORT = "SrtListenPort";
//
void to_json(njson& j, const CGICmd::st_Srt& p) {
    j[SRT_ENCRYPTION] = p.SrtEncryption;
    j[SRT_PASSPHRASE] = p.SrtPassphrase;
    j[SRT_PASSPHRASE_USED] = p.SrtPassphraseUsed;
    //
    j[SRT_LISTEN_PORT] = std::to_string(p.SrtListenPort);
}
void from_json(const njson& j, CGICmd::st_Srt& p) {
    json_get_val(j, SRT_ENCRYPTION, p.SrtEncryption);
    json_get_val(j, SRT_PASSPHRASE, p.SrtPassphrase);
    json_get_val(j, SRT_PASSPHRASE_USED, p.SrtPassphraseUsed);
    //
    conv_json_str2num(j, SRT_LISTEN_PORT, p.SrtListenPort);
}

}   // namespace CGICmd.
