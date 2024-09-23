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
#include "JSON_utils.h"
#include "RingBuffer.h"
#include "CGI_cmd_parameter.h"

namespace CGICmd
{

namespace COMMON
{

enum em_OnOff
{
    OFF,
    ON,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_OnOff, {
    {OFF, "off"},
    {ON, "on"},
})

enum em_TrueFalse
{
    FALSE = 0,
    TRUE = 1,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_TrueFalse, {
    {FALSE, "0"},
    {TRUE, "1"},
})

enum em_EnableDisable
{
    ENABLE,
    DISABLE,
    DISPLAY_ONLY,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_EnableDisable, {
    {ENABLE, "enable"},
    {DISABLE, "disable"},
    {DISPLAY_ONLY, "display_only"},
})

enum em_LowHigh
{
    LOW,
    HIGH,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_LowHigh, {
    {LOW, "low"},
    {HIGH, "high"},
})

enum em_PressRelease
{
    PRESS,
    RELEASE,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_PressRelease, {
    {PRESS, "press"},
    {RELEASE, "release"},
})

}   // namespace COMMON.



struct st_Param2
{
    union {
        int v1;
        int min;
        int x;
    };
    union {
        int v2;
        int max;
        int y;
    };
};
using st_Range = st_Param2;
using st_Position = st_Param2;

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

inline auto conv_param22str = [](const st_Param2 &prm2) -> std::string {
    std::stringstream ss;
    ss << prm2.v1 << "," << prm2.v2;
    std::string param2_str = ss.str();

    return param2_str;
};
inline auto conv_range2str = [](const st_Range &range) -> std::string {
    return conv_param22str(range);
};
inline auto conv_position2str = [](const st_Position &pos) -> std::string {
    return conv_param22str(pos);
};

inline auto conv_json_str2param2 = [](const njson& j, const char *key, st_Param2 &prm2) -> bool {
    std::string param2_str;
    json_get_val(j, key, param2_str);
    std::stringstream ss{param2_str};
    std::string str;
    std::vector<std::string> v;
    while (std::getline(ss, str, ',')) v.push_back(str);

    bool ret = false;
    if (v.size() >= 2) {
        ret = conv_str2num(v[0], prm2.v1) && conv_str2num(v[1], prm2.v2);
    }

    return ret;
};
inline auto conv_json_str2range = [](const njson& j, const char *key, st_Range &range) -> bool {
    return conv_json_str2param2(j, key, range);
};
inline auto conv_json_str2position = [](const njson& j, const char *key, st_Position &pos) -> bool {
    return conv_json_str2param2(j, key, pos);
};

template<typename T> struct st_List
{
    std::vector<T> buf;
};

inline auto conv_list2str = []<class T>(const st_List<T> &lst) -> std::string {
    std::stringstream ss;
    for (auto &e : lst.buf) {
        auto str = json_conv_enum2str(e);
        ss << str;
        ss << ",";
    }
    auto str = ss.str();
    if (!str.empty() && str.back() == ',') str.pop_back();

    return str;
};

inline auto conv_json_str2list = []<class T>(const njson& j, const char *key, st_List<T> &lst) -> bool {
    std::string list_str;
    json_get_val(j, key, list_str);
    std::stringstream ss{list_str};
    std::string str;
    std::vector<std::string> v;
    while (std::getline(ss, str, ',')) v.push_back(str);

    lst.buf.clear();
    for (auto &e : v) {
        njson j = e;
        T val = j.template get<T>();
        lst.buf.push_back(val);
    }

    return true;
};




/////////////////////////////////////////////////////////////////////
// system.
enum em_Power
{
    Power_ON,
    Power_STANDBY,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_Power, {
    {Power_ON, "on"},
    {Power_STANDBY, "standby"},
})

enum em_LensCalibrateStatus
{
    LensCalibrateStatus_CALIBRATED,
    LensCalibrateStatus_NOT_CALIBRATED,
    LensCalibrateStatus_NO_MOUNT,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_LensCalibrateStatus, {
    {LensCalibrateStatus_CALIBRATED, "calibrated"},
    {LensCalibrateStatus_NOT_CALIBRATED, "not_calibrated"},
    {LensCalibrateStatus_NO_MOUNT, "no_mount"},
})

struct st_System
{
    static constexpr auto cmd = "system";

    em_Power Power;

    std::string CGIVersion;
    std::string ModelName;
    std::string Serial;
    std::string SoftVersion;

    std::string LensModelName;
    std::string LensSerial;
    std::string LensSoftVersion;

    em_LensCalibrateStatus LensCalibrateStatus;

public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        st_System,

        Power,

        CGIVersion,
        ModelName,
        Serial,
        SoftVersion,

        LensModelName,
        LensSerial,
        LensSoftVersion,

        LensCalibrateStatus
    )
};



/////////////////////////////////////////////////////////////////////
// status.
struct st_Status
{
    static constexpr auto cmd = "status";

    COMMON::em_TrueFalse TemperatureWarning;

public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        st_Status,

        TemperatureWarning
    )
};



/////////////////////////////////////////////////////////////////////
// ptzf.
enum em_AutoManual
{
    AUTO,
    MANUAL,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_AutoManual, {
    {AUTO, "auto"},
    {MANUAL, "manual"},
})

enum em_TouchFocusInMF
{
    TRACKING_AF,
    SPOT_FOCUS,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_TouchFocusInMF, {
    {TRACKING_AF, "tracking_af"},
    {SPOT_FOCUS, "spot_focus"},
})

struct st_Ptzf
{
    static constexpr auto cmd = "ptzf";

    static constexpr auto POS_X_MAX = 640 - 1;
    static constexpr auto POS_Y_MAX = 480 - 1;

    em_AutoManual FocusMode;
    COMMON::em_PressRelease FocusTrackingCancel;
    st_Position FocusTrackingPosition;
    em_TouchFocusInMF TouchFocusInMF;

public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        st_Ptzf,

        FocusMode,
        TouchFocusInMF
    )
};



/////////////////////////////////////////////////////////////////////
// imaging.
/////////////////////////////////////////////////////////////////
// shutter.
enum em_ExposureShutterMode
{
    ExposureShutterMode_SPEED,
    ExposureShutterMode_ANGLE,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_ExposureShutterMode, {
    {ExposureShutterMode_SPEED, "speed"},
    {ExposureShutterMode_ANGLE, "angle"},
})

enum em_ExposureShutterModeState
{
    ExposureShutterModeState_OFF,
    ExposureShutterModeState_SPEED,
    ExposureShutterModeState_ANGLE,
    ExposureShutterModeState_ECS,
    ExposureShutterModeState_AUTO,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_ExposureShutterModeState, {
    {ExposureShutterModeState_OFF, "off"},
    {ExposureShutterModeState_SPEED, "speed"},
    {ExposureShutterModeState_ANGLE, "angle"},
    {ExposureShutterModeState_ECS, "ecs"},
    {ExposureShutterModeState_AUTO, "auto"},
})

/////////////////////////////////////////////////////////////////
// white balance.
enum em_WhiteBalanceGainTemp
{
    WhiteBalanceGainTemp_GAIN,
    WhiteBalanceGainTemp_TEMP,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_WhiteBalanceGainTemp, {
    {WhiteBalanceGainTemp_GAIN, "gain"},
    {WhiteBalanceGainTemp_TEMP, "temp"},
})

enum em_WhiteBalanceMode
{
    WhiteBalanceMode_ATW,
    WhiteBalanceMode_MEMORY_A,
    WhiteBalanceMode_PRESET,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_WhiteBalanceMode, {
    {WhiteBalanceMode_ATW, "atw"},
    {WhiteBalanceMode_MEMORY_A, "memory_a"},
    {WhiteBalanceMode_PRESET, "preset"},
})

enum class em_WhiteBalanceModeState
{
    AUTO,
    MANUAL,
    INVALID,
};

/////////////////////////////////////////////////////////////////
// ISO.
enum em_ExposureBaseISO
{
    ExposureBaseISO_ISO800,
    ExposureBaseISO_ISO12800,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_ExposureBaseISO, {
    {ExposureBaseISO_ISO800, "iso800"},
    {ExposureBaseISO_ISO12800, "iso12800"},
})

enum em_ExposureISOGainMode
{
    ExposureISOGainMode_ISO,
    ExposureISOGainMode_GAIN,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_ExposureISOGainMode, {
    {ExposureISOGainMode_ISO, "iso"},
    {ExposureISOGainMode_GAIN, "gain"},
})

enum class em_ISOModeState
{
    GAIN,
    ISO,
    CINE_EI_QUITCK,
    CINE_EI,
    INVALID,
};

/////////////////////////////////////////////////////////////////
// IRIS.
enum class em_IrisModeState
{
    AUTO,
    MANUAL,
    INVALID,
};

/////////////////////////////////////////////////////////////////
// ND.
enum em_ExposureNDClear
{
    ExposureNDClear_FILTERED,
    ExposureNDClear_CLEAR,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_ExposureNDClear, {
    {ExposureNDClear_FILTERED, "filtered"},
    {ExposureNDClear_CLEAR, "clear"},
})

enum class em_NDModeState
{
    AUTO,
    MANUAL,
    CLEAR,
    INVALID,
};



struct st_Imaging
{
    static constexpr auto cmd = "imaging";

    /////////////////////////////////////////////////////////////////
    // shutter.
    int ExposureAngle;
    st_Range ExposureAngleRange;

    int ExposureECS;
    st_Range ExposureECSRange;
    int ExposureECSValue;   // 0.001 ECS.

    int ExposureExposureTime;
    st_Range ExposureExposureTimeRange;

    em_ExposureShutterModeState ExposureShutterModeState;

    /////////////////////////////////////////////////////////////////
    // white balance.
    int WhiteBalanceColorTemp;  // 2000...15000 [K].
    int WhiteBalanceColorTempCurrent;

    em_WhiteBalanceMode WhiteBalanceMode;
    em_WhiteBalanceGainTemp WhiteBalanceGainTemp;

    /////////////////////////////////////////////////////////////////
    // ISO.
    COMMON::em_OnOff ExposureAGCEnable;
    em_ExposureBaseISO ExposureBaseISO;
    COMMON::em_EnableDisable ExposureBaseISOPmt;
    COMMON::em_LowHigh ExposureBaseSensitivity;
    int ExposureExposureIndex;
    COMMON::em_EnableDisable ExposureExposureIndexPmt;
    int ExposureGain;
    int ExposureGainTemporary;
    int ExposureISO;
    int ExposureISOTemporary;
    em_ExposureISOGainMode ExposureISOGainMode;

    /////////////////////////////////////////////////////////////////
    // IRIS.
    COMMON::em_OnOff ExposureAutoIris;
    int ExposureFNumber;    // 0.01/digit.
    int ExposureIris;   // inquiry -> value is [256 / 3 = 85.3... step]
    st_Range ExposureIrisRange;

    /////////////////////////////////////////////////////////////////
    // ND.
    COMMON::em_OnOff ExposureAutoNDFilterEnable;
    em_ExposureNDClear ExposureNDClear;
    int ExposureNDVariable;
};



/////////////////////////////////////////////////////////////////////
// cameraoperation.
enum em_MediaRecordingStatus
{
    MediaRecordingStatus_STANDBY,
    MediaRecordingStatus_REC,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_MediaRecordingStatus, {
    {MediaRecordingStatus_STANDBY, "standby"},
    {MediaRecordingStatus_REC, "rec"},
})

struct st_Cameraoperation
{
    static constexpr auto cmd = "cameraoperation";

    COMMON::em_PressRelease MediaRecording;
    em_MediaRecordingStatus MediaRecordingStatus;

public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        st_Cameraoperation,

        MediaRecordingStatus
    )
};



/////////////////////////////////////////////////////////////////////
// tally.
enum em_TurnOnOff
{
    TURN_OFF,
    TURN_ON,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_TurnOnOff, {
    {TURN_OFF, "turn_off"},
    {TURN_ON, "turn_on"},
})

enum em_InternalExternal
{
    INTERNAL,
    EXTERNAL,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_InternalExternal, {
    {INTERNAL, "internal"},
    {EXTERNAL, "external"},
})

enum em_OffLowHigh
{
    OFF,
    LOW,
    HIGH,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_OffLowHigh, {
    {OFF, "off"},
    {LOW, "low"},
    {HIGH, "high"},
})

struct st_Tally
{
    static constexpr auto cmd = "tally";

    COMMON::em_OnOff GTallyLampEnable;
    em_TurnOnOff GTallyControl;
    em_InternalExternal TallyControlMode;
    em_TurnOnOff RTallyControl;
    em_OffLowHigh TallyLampBrightness;
    COMMON::em_OnOff TallyLampBrightnessExtra;

public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        st_Tally,

        GTallyLampEnable,
        GTallyControl,
        TallyControlMode,
        RTallyControl,
        TallyLampBrightness,
        TallyLampBrightnessExtra
    )
};



/////////////////////////////////////////////////////////////////////
// project.
enum em_BaseSettingShootingMode
{
    BaseSettingShootingMode_CUSTOM,
    BaseSettingShootingMode_FLEXIBLE_ISO,
    BaseSettingShootingMode_CINE_EI,
    BaseSettingShootingMode_CINE_EI_QUICK,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_BaseSettingShootingMode, {
    {BaseSettingShootingMode_CUSTOM, "custom"},
    {BaseSettingShootingMode_FLEXIBLE_ISO, "flexible_iso"},
    {BaseSettingShootingMode_CINE_EI, "cine_ei"},
    {BaseSettingShootingMode_CINE_EI_QUICK, "cine_ei_quick"},
})

enum em_RecFormatFrequency
{
    RecFormatFrequency_5994,
    RecFormatFrequency_5000,
    RecFormatFrequency_2997,
    RecFormatFrequency_2500,
    RecFormatFrequency_2400,
    RecFormatFrequency_2398,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_RecFormatFrequency, {
    {RecFormatFrequency_5994, "5994"},
    {RecFormatFrequency_5000, "5000"},
    {RecFormatFrequency_2997, "2997"},
    {RecFormatFrequency_2500, "2500"},
    {RecFormatFrequency_2400, "2400"},
    {RecFormatFrequency_2398, "2398"},
})

enum em_RecFormatVideoFormat
{
    RecFormatVideoFormat_4096x2160p,
    RecFormatVideoFormat_3840x2160p,
    RecFormatVideoFormat_1920x1080p,
    RecFormatVideoFormat_1920x1080p_50,
    RecFormatVideoFormat_1920x1080p_35,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_RecFormatVideoFormat, {
    {RecFormatVideoFormat_4096x2160p, "4096x2160p"},
    {RecFormatVideoFormat_3840x2160p, "3840x2160p"},
    {RecFormatVideoFormat_1920x1080p, "1920x1080p"},
    {RecFormatVideoFormat_1920x1080p_50, "1920x1080p_50"},
    {RecFormatVideoFormat_1920x1080p_35, "1920x1080p_35"},
})

struct st_Project
{
    static constexpr auto cmd = "project";

    em_BaseSettingShootingMode BaseSettingShootingMode;
    em_RecFormatFrequency RecFormatFrequency;
    em_RecFormatVideoFormat RecFormatVideoFormat;
    st_List<em_RecFormatVideoFormat> RecFormatVideoFormatList;
};



/////////////////////////////////////////////////////////////////
// network.
struct st_Network
{
    static constexpr auto cmd = "network";

    int HttpPort;
    std::string Hostname;
    std::string MacAddress;
    std::string CameraName;
};



/////////////////////////////////////////////////////////////////
// stream.
enum em_StreamMode
{
    StreamMode_RTSP,
    StreamMode_RTMP,
    StreamMode_SRT_CALLER,
    StreamMode_SRT_LISTENER,
    StreamMode_NDI_HX,
    StreamMode_OFF,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_StreamMode, {
    {StreamMode_RTSP, "rtsp"},
    {StreamMode_RTMP, "rtmp"},
    {StreamMode_SRT_CALLER, "srt-caller"},
    {StreamMode_SRT_LISTENER, "srt-listener"},
    {StreamMode_NDI_HX, "ndi_hx"},
    {StreamMode_OFF, "off"},
})

enum em_StreamStatus
{
    StreamStatus_INVALID,
    StreamStatus_OFF,
    StreamStatus_READY,
    StreamStatus_READY_SSL,
    StreamStatus_STREAMING,
    StreamStatus_STREAMING_SSL,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_StreamStatus, {
    {StreamStatus_INVALID, "invalid"},
    {StreamStatus_OFF, "off"},
    {StreamStatus_READY, "ready"},
    {StreamStatus_READY_SSL, "ready-ssl"},
    {StreamStatus_STREAMING, "streaming"},
    {StreamStatus_STREAMING_SSL, "streaming-ssl"},
})

struct st_Stream
{
    static constexpr auto cmd = "stream";

    em_StreamMode StreamMode;
    em_StreamStatus StreamStatus;

public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        st_Stream,

        StreamMode,
        StreamStatus
    )
};



/////////////////////////////////////////////////////////////////
// srt.
enum em_SrtEncryption
{
    SrtEncryption_NONE,
    SrtEncryption_AES_128,
    SrtEncryption_AES_256,
};
NLOHMANN_JSON_SERIALIZE_ENUM( em_SrtEncryption, {
    {SrtEncryption_NONE, "none"},
    {SrtEncryption_AES_128, "aes-128"},
    {SrtEncryption_AES_256, "aes-256"},
})

struct st_Srt
{
    static constexpr auto cmd = "srt";

    em_SrtEncryption SrtEncryption;
    std::string SrtPassphrase;
    COMMON::em_TrueFalse SrtPassphraseUsed;

    int SrtListenPort;
};



void to_json(njson& j, const st_Imaging& p);
void from_json(const njson& j, st_Imaging& p);

void to_json(njson& j, const st_Project& p);
void from_json(const njson& j, st_Project& p);

void to_json(njson& j, const st_Network& p);
void from_json(const njson& j, st_Network& p);

void to_json(njson& j, const st_Srt& p);
void from_json(const njson& j, st_Srt& p);

}   // namespace CGICmd.



class CGI
{
private:
    // HTTP
    cpr::Url cgi_url;
	cpr::Authentication cgi_auth;
	cpr::Header cgi_header;

    bool connected;

    // command.
    bool auth;
    bool timeout;
    struct st_CmdInfo
    {
        CGICmd::st_System system;
        CGICmd::st_Status status;
        CGICmd::st_Ptzf ptzf;
        CGICmd::st_Imaging imaging;
        CGICmd::st_Cameraoperation cameraoperation;
        CGICmd::st_Tally tally;
        CGICmd::st_Project project;
        CGICmd::st_Network network;
        CGICmd::st_Stream stream;
        CGICmd::st_Srt srt;
    };

    std::mutex mtx;
    std::mutex mtx_auth;

    bool running;

    StopWatch sw_inq, sw_set;
    double lap_cur_inq, lap_cur_set;
    double lap_ave_inq, lap_ave_set;

    st_CmdInfo cmd_info;
    std::unique_ptr<RingBufferAsync<st_CmdInfo>> cmd_info_list;
    std::atomic_bool is_update_cmd_info_list;

    std::unique_ptr<RingBufferAsync<std::string>> cmd_msg_list;
    std::mutex mtx_cmd_msg;
    std::condition_variable cv_cmd_msg;
    std::atomic_bool is_update_cmd_msg_list;
    std::atomic_bool is_send_set_cmd;

    std::string make_referer_message(const std::string &server_adr, int server_port)
    {
        std::stringstream ss;
        ss << "http://"
            << server_adr
            << ":" << server_port
            << "/";
        std::string str = ss.str();

        return str;
    }

public:
    CGI() = delete;
    CGI(const std::string &ip_address, int port, const std::string &id = "", const std::string &pw = "")
        : cgi_url(cpr::Url{ ip_address + std::to_string(port) })
        , cgi_auth(cpr::Authentication{ id, pw, cpr::AuthMode::DIGEST })
    {
        cgi_url = cpr::Url{ ip_address + ":" + std::to_string(port) };

        connected = false;

        set_referer_message(ip_address, port);

        auth = false;
        timeout = false;

        running = true;

        lap_cur_inq = 1.0;
        lap_ave_inq = 1.0;
        lap_cur_set = 1.0;
        lap_ave_set = 1.0;

        constexpr auto RING_BUF_SZ = 1;
        cmd_info = {};
        cmd_info_list = std::make_unique<RingBufferAsync<st_CmdInfo>>(RING_BUF_SZ);
        is_update_cmd_info_list.store(false);

        constexpr auto CMD_MSG_BUF_SZ = 1;
        cmd_msg_list = std::make_unique<RingBufferAsync<std::string>>(CMD_MSG_BUF_SZ, false, false);
        is_update_cmd_msg_list.store(false);
        is_send_set_cmd.store(false);
    }

    virtual ~CGI() {}

    bool is_running() const { return running; }
    void start() { running = true; }
    void stop() { running = false; }

    bool is_connected() const { return connected; }
    bool is_auth() const { return auth; }
    bool is_timeout() const { return timeout; }

    bool empty_cmd_msg_list() const { return cmd_msg_list->empty(); }
    bool full_cmd_msg_list() const { return cmd_msg_list->full(); }

    std::tuple<double, double> get_lap_inq() { return { lap_cur_inq, lap_ave_inq }; }
    std::tuple<double, double> get_lap_set() { return { lap_cur_set, lap_ave_set }; }

    // auto &get_cmd_info() { return cmd_info; }
    // auto &get_cmd_info() const { return cmd_info; }

    void set_account(const std::string &id, const std::string &pw)
    {
        std::lock_guard<std::mutex> lg(mtx_auth);

        cgi_auth = cpr::Authentication{ id, pw, cpr::AuthMode::DIGEST };
        auth = true;
    }

    void set_referer_message(const std::string &server_adr, int server_port)
    {
        auto referer_str = make_referer_message(server_adr, server_port);

        cgi_header = cpr::Header{
            { "Referer", referer_str },
        };

    }

    std::tuple<long, std::string, std::string> GET(const std::string &message)
    {
        std::lock_guard<std::mutex> lg(mtx_auth);

        if (!auth) return { cpr::status::HTTP_UNAUTHORIZED, "", "" };

		auto ar = cpr::GetAsync(cgi_url + message, cgi_auth, cgi_header);
        constexpr auto TIMEOUT_MS = 1000;
        auto wait_status = ar.wait_for(std::chrono::milliseconds(TIMEOUT_MS));

        if (wait_status == std::future_status::timeout) {
            timeout = true;
            return { 0, "", "" };
        } else {
            timeout = false;
        }

        auto res = ar.get();

        if (res.error.code == cpr::ErrorCode::CONNECTION_FAILURE) {
            connected = false;
            return { 0, "", "" };
        } else {
            connected = true;
        }

        auto status = res.status_code;
        auto reson = res.reason;
        auto body = res.text;

        if (status == cpr::status::HTTP_UNAUTHORIZED) auth = false;

        return { status, reson, body };
    }

    template<typename T> void set_command(const std::string &param, const std::string &val)
    {
        std::stringstream ss;
        ss << "/command/" << T::cmd << ".cgi?" << param << "=" << val;
        std::string msg = ss.str();

        cmd_msg_list->Write(msg);
        is_update_cmd_msg_list.store(true);
        cv_cmd_msg.notify_one();
    }

    template<typename T> void set_command(const std::string &param_list)
    {
        std::stringstream ss;
        ss << "/command/" << T::cmd << ".cgi?" << param_list;
        std::string msg = ss.str();

        cmd_msg_list->Write(msg);
        is_update_cmd_msg_list.store(true);
        cv_cmd_msg.notify_one();
    }



    /////////////////////////////////////////////////////////////////
    // ptzf.
    void set_ptzf_FocusMode(CGICmd::em_AutoManual val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "FocusMode=" + str;
        set_command<CGICmd::st_Ptzf>(msg);
    }

    void click_ptzf_FocusTrackingCancel()
    {
        std::string msg;

        // press.
        msg = "FocusTrackingCancel=" + json_conv_enum2str(CGICmd::COMMON::PRESS);
        set_command<CGICmd::st_Ptzf>(msg);

        // wait for sending 'press' command.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // release.
        msg = "FocusTrackingCancel=" + json_conv_enum2str(CGICmd::COMMON::RELEASE);
        set_command<CGICmd::st_Ptzf>(msg);
    }

    void set_ptzf_FocusTrackingPosition(int x, int y)
    {
        std::string msg;
        x = std::clamp(x, 0, CGICmd::st_Ptzf::POS_X_MAX);
        y = std::clamp(y, 0, CGICmd::st_Ptzf::POS_Y_MAX);
        CGICmd::st_Position pos{x, y};
        auto str = CGICmd::conv_position2str(pos);
        msg = "FocusTrackingPosition=" + str;
        set_command<CGICmd::st_Ptzf>(msg);
    }

    void set_ptzf_TouchFocusInMF(CGICmd::em_TouchFocusInMF val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "TouchFocusInMF=" + str;
        set_command<CGICmd::st_Ptzf>(msg);
    }



    /////////////////////////////////////////////////////////////////
    // shutter.
    // Auto:  /command/imaging.cgi?ExposureAutoShutterEnable=on
    // Speed: /command/imaging.cgi?ExposureShutterMode=speed&ExposureAutoShutterEnable=off&ExposureShutterSpeedEnable=on&ExposureECSEnable=off
    // Angle: /command/imaging.cgi?ExposureShutterMode=angle&ExposureAutoShutterEnable=off&ExposureECSEnable=off
    // ECS:   /command/imaging.cgi?ExposureShutterMode=angle&ExposureAutoShutterEnable=off&ExposureECSEnable=on
    //        /command/imaging.cgi?ExposureShutterMode=speed&ExposureAutoShutterEnable=off&ExposureShutterSpeedEnable=on&ExposureECSEnable=on
    // Off:   /command/imaging.cgi?ExposureShutterMode=speed&ExposureAutoShutterEnable=off&ExposureShutterSpeedEnable=off
    void set_imaging_ExposureShutterModeState(CGICmd::em_ExposureShutterModeState state)
    {
        std::string msg;
        switch (state) {
        case CGICmd::ExposureShutterModeState_AUTO:
            msg = "ExposureAutoShutterEnable=on";
            break;
        case CGICmd::ExposureShutterModeState_SPEED:
            msg = "ExposureShutterMode=speed&ExposureAutoShutterEnable=off&ExposureShutterSpeedEnable=on&ExposureECSEnable=off";
            break;
        case CGICmd::ExposureShutterModeState_ANGLE:
            msg = "ExposureShutterMode=angle&ExposureAutoShutterEnable=off&ExposureECSEnable=off";
            break;
        case CGICmd::ExposureShutterModeState_ECS:
            msg = "ExposureShutterMode=speed&ExposureAutoShutterEnable=off&ExposureShutterSpeedEnable=on&ExposureECSEnable=on";
            break;
        case CGICmd::ExposureShutterModeState_OFF:
            msg = "ExposureShutterMode=speed&ExposureAutoShutterEnable=off&ExposureShutterSpeedEnable=off";
            break;
        default:
            return;
        }
        set_command<CGICmd::st_Imaging>(msg);
    }

    void set_imaging_ExposureExposureTime(int idx)
    {
        std::string msg;
        msg = "ExposureExposureTime=" + std::to_string(idx);
        set_command<CGICmd::st_Imaging>(msg);
    }

    void set_imaging_ExposureAngle(int idx)
    {
        std::string msg;
        msg = "ExposureAngle=" + std::to_string(idx);
        set_command<CGICmd::st_Imaging>(msg);
    }

    void set_imaging_ExposureECS(int idx)
    {
        std::string msg;
        msg = "ExposureECS=" + std::to_string(idx);
        set_command<CGICmd::st_Imaging>(msg);
    }



    /////////////////////////////////////////////////////////////////
    // white balance.
    // ATW:           /command/imaging.cgi?WhiteBalanceMode=atw&WhiteBalanceOffsetColorTemp=0&WhiteBalanceOffsetTint=0
    // memory_A(T/T): /command/imaging.cgi?WhiteBalanceMode=memory_a&WhiteBalanceGainTemp=temp&WhiteBalanceTint=0
    // memory_A(R/B): /command/imaging.cgi?WhiteBalanceMode=memory_a&WhiteBalanceGainTemp=gain
    // preset:        /command/imaging.cgi?WhiteBalanceMode=preset
    void set_imaging_WhiteBalanceModeState(CGICmd::em_WhiteBalanceModeState state)
    {
        std::string msg;
        switch (state) {
        case CGICmd::em_WhiteBalanceModeState::AUTO:
            msg = "WhiteBalanceMode=atw&WhiteBalanceOffsetColorTemp=0&WhiteBalanceOffsetTint=0";
            break;
        case CGICmd::em_WhiteBalanceModeState::MANUAL:
            msg = "WhiteBalanceMode=memory_a&WhiteBalanceGainTemp=temp&WhiteBalanceTint=0";
            break;
        default:
            return;
        }
        set_command<CGICmd::st_Imaging>(msg);
    }

    void set_imaging_WhiteBalanceColorTemp(int idx)
    {
        std::string msg;
        msg = "WhiteBalanceColorTemp=" + std::to_string(idx);
        set_command<CGICmd::st_Imaging>(msg);
    }

    CGICmd::em_WhiteBalanceModeState get_wb_mode_state() const
    {
        using state_t = CGICmd::em_WhiteBalanceModeState;
        state_t state = state_t::INVALID;

        auto mode = cmd_info.imaging.WhiteBalanceMode;
        auto gain_temp = cmd_info.imaging.WhiteBalanceGainTemp;

        if (mode == CGICmd::WhiteBalanceMode_ATW) {
            state = state_t::AUTO;
        } else if (mode == CGICmd::WhiteBalanceMode_MEMORY_A
            && gain_temp == CGICmd::WhiteBalanceGainTemp_TEMP
            ) {
            state = state_t::MANUAL;
        }

        return state;
    }

    void set_wb_mode_state(CGICmd::em_WhiteBalanceModeState state)
    {
        using state_t = CGICmd::em_WhiteBalanceModeState;

        auto &mode = cmd_info.imaging.WhiteBalanceMode;
        auto &gain_temp = cmd_info.imaging.WhiteBalanceGainTemp;

        switch (state) {
        case state_t::MANUAL:
            mode = CGICmd::WhiteBalanceMode_MEMORY_A;
            gain_temp = CGICmd::WhiteBalanceGainTemp_TEMP;
            break;

        case state_t::AUTO:
        case state_t::INVALID:
        default:
            mode = CGICmd::WhiteBalanceMode_ATW;
        }
    }



    /////////////////////////////////////////////////////////////////
    // ISO.
    void set_imaging_ExposureAGCEnable(CGICmd::COMMON::em_OnOff val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "ExposureAGCEnable=" + str;
        set_command<CGICmd::st_Imaging>(msg);
    }

    void set_imaging_ExposureBaseSensitivity(CGICmd::COMMON::em_LowHigh val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "ExposureBaseSensitivity=" + str;
        set_command<CGICmd::st_Imaging>(msg);
    }

    void set_imaging_ExposureGain(int val)
    {
        std::string msg;
        msg = "ExposureGain=" + std::to_string(val);
        set_command<CGICmd::st_Imaging>(msg);
    }

    void set_imaging_ExposureBaseISO(CGICmd::em_ExposureBaseISO val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "ExposureBaseISO=" + str;
        set_command<CGICmd::st_Imaging>(msg);
    }

    void set_imaging_ExposureISO(int val)
    {
        std::string msg;
        msg = "ExposureISO=" + std::to_string(val);
        set_command<CGICmd::st_Imaging>(msg);
    }

    void set_imaging_ExposureExposureIndex(int val)
    {
        std::string msg;
        msg = "ExposureExposureIndex=" + std::to_string(val);
        set_command<CGICmd::st_Imaging>(msg);
    }

    CGICmd::em_ISOModeState get_iso_mode_state() const
    {
        using state_t = CGICmd::em_ISOModeState;
        state_t state = state_t::INVALID;

        auto idx_pmt = cmd_info.imaging.ExposureExposureIndexPmt;   // Cine EI ?
        auto iso_pmt = cmd_info.imaging.ExposureBaseISOPmt;         // Cine EI / Cine EI Quick ?
        auto iso_gain = cmd_info.imaging.ExposureISOGainMode;       // ISO / Gain ?

        if (idx_pmt == CGICmd::COMMON::ENABLE) {
            if (iso_pmt == CGICmd::COMMON::ENABLE) {
                state = state_t::CINE_EI;
            } else if (iso_pmt == CGICmd::COMMON::DISPLAY_ONLY) {
                state = state_t::CINE_EI_QUITCK;
            }
        } else {
            if (iso_gain == CGICmd::ExposureISOGainMode_ISO) {
                state = state_t::ISO;
            } else {
                state = state_t::GAIN;
            }
        }

        return state;
    }



    /////////////////////////////////////////////////////////////////
    // IRIS.
    void set_imaging_ExposureAutoIris(CGICmd::COMMON::em_OnOff val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "ExposureAutoIris=" + str;
        set_command<CGICmd::st_Imaging>(msg);
    }

    void set_imaging_ExposureIris(int val)
    {
        std::string msg;
        msg = "ExposureIris=" + std::to_string(val);
        set_command<CGICmd::st_Imaging>(msg);
    }

    CGICmd::em_IrisModeState get_iris_mode_state() const
    {
        using state_t = CGICmd::em_IrisModeState;
        state_t state = state_t::INVALID;

        auto is_auto = cmd_info.imaging.ExposureAutoIris;

        if (is_auto == CGICmd::COMMON::ON) {
            state = state_t::AUTO;
        } else if (is_auto == CGICmd::COMMON::OFF) {
            state = state_t::MANUAL;
        }

        return state;
    }



    /////////////////////////////////////////////////////////////////
    // ND.
    // void set_imaging_ExposureAutoNDFilterEnable(CGICmd::COMMON::em_OnOff val)
    // {
    //     std::string msg;
    //     auto str = json_conv_enum2str(val);
    //     msg = "ExposureAutoNDFilterEnable=" + str;
    //     set_command<CGICmd::st_Imaging>(msg);
    // }

    // void set_imaging_ExposureNDClear(CGICmd::em_ExposureNDClear val)
    // {
    //     std::string msg;
    //     auto str = json_conv_enum2str(val);
    //     msg = "ExposureNDClear=" + str;
    //     set_command<CGICmd::st_Imaging>(msg);
    // }

    void set_imaging_ExposureNDVariable(int val)
    {
        std::string msg;
        msg = "ExposureNDVariable=" + std::to_string(val);
        set_command<CGICmd::st_Imaging>(msg);
    }

    void set_nd_mode_state(CGICmd::em_NDModeState state)
    {
        using state_t = CGICmd::em_NDModeState;

        std::string msg;
        switch (state) {
        case state_t::AUTO:
            {
                auto str_nd_clear = json_conv_enum2str(CGICmd::ExposureNDClear_FILTERED);
                auto str_nd_auto = json_conv_enum2str(CGICmd::COMMON::ON);
                msg = "ExposureNDClear=" + str_nd_clear + "&" + "ExposureAutoNDFilterEnable=" + str_nd_auto;
            }
            break;
 
        case state_t::MANUAL:
            {
                auto str_nd_clear = json_conv_enum2str(CGICmd::ExposureNDClear_FILTERED);
                auto str_nd_auto = json_conv_enum2str(CGICmd::COMMON::OFF);
                msg = "ExposureNDClear=" + str_nd_clear + "&" + "ExposureAutoNDFilterEnable=" + str_nd_auto;
            }
            break;
 
        case state_t::CLEAR:
            {
                auto str_nd_clear = json_conv_enum2str(CGICmd::ExposureNDClear_CLEAR);
                msg = "ExposureNDClear=" + str_nd_clear;
            }
            break;
 
        case state_t::INVALID:
        default:
            return;
        }

        set_command<CGICmd::st_Imaging>(msg);
    }

    CGICmd::em_NDModeState get_nd_mode_state() const
    {
        using state_t = CGICmd::em_NDModeState;
        state_t state = state_t::INVALID;

        auto is_auto = (cmd_info.imaging.ExposureAutoNDFilterEnable == CGICmd::COMMON::ON);
        auto is_clear = (cmd_info.imaging.ExposureNDClear == CGICmd::ExposureNDClear_CLEAR);

        if (is_clear) {
            state = state_t::CLEAR;
        } else if (is_auto) {
            state = state_t::AUTO;
        } else {
            state = state_t::MANUAL;
        }

        return state;
    }



    /////////////////////////////////////////////////////////////////
    // cameraoperation.
    void click_cameraoperation_MediaRecording()
    {
        std::string msg;

        // press.
        msg = "MediaRecording=" + json_conv_enum2str(CGICmd::COMMON::PRESS);
        set_command<CGICmd::st_Cameraoperation>(msg);

        // wait for sending 'press' command.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // release.
        msg = "MediaRecording=" + json_conv_enum2str(CGICmd::COMMON::RELEASE);
        set_command<CGICmd::st_Cameraoperation>(msg);
    }



    /////////////////////////////////////////////////////////////////
    // tally.
    void set_tally_GTallyLampEnable(CGICmd::COMMON::em_OnOff val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "GTallyLampEnable=" + str;
        set_command<CGICmd::st_Tally>(msg);
    }

    void set_tally_GTallyControl(CGICmd::em_TurnOnOff val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "GTallyControl=" + str;
        set_command<CGICmd::st_Tally>(msg);
    }

    void set_tally_TallyControlMode(CGICmd::em_InternalExternal val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "TallyControlMode=" + str;
        set_command<CGICmd::st_Tally>(msg);
    }

    void set_tally_RTallyControl(CGICmd::em_TurnOnOff val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "RTallyControl=" + str;
        set_command<CGICmd::st_Tally>(msg);
    }

    void set_tally_TallyLampBrightness(CGICmd::em_OffLowHigh val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "TallyLampBrightness=" + str;
        set_command<CGICmd::st_Tally>(msg);
    }

    void set_tally_TallyLampBrightnessExtra(CGICmd::COMMON::em_OnOff val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "TallyLampBrightnessExtra=" + str;
        set_command<CGICmd::st_Tally>(msg);
    }

    void set_comb_tally(CGICmd::COMMON::em_OnOff g_lamp_enable, CGICmd::em_InternalExternal mode, CGICmd::em_OffLowHigh brightness)
    {
        std::string msg;
        msg = "GTallyLampEnable=" + json_conv_enum2str(g_lamp_enable);
        msg += "&";
        msg += "TallyControlMode=" + json_conv_enum2str(mode);
        msg += "&";
        msg += "TallyLampBrightness=" + json_conv_enum2str(brightness);
        set_command<CGICmd::st_Tally>(msg);
    }



    /////////////////////////////////////////////////////////////////
    // project.
    void set_project_RecFormatFrequency(CGICmd::em_RecFormatFrequency val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "RecFormatFrequency=" + str;
        set_command<CGICmd::st_Project>(msg);
    }

    void set_project_BaseSettingShootingMode(CGICmd::em_BaseSettingShootingMode val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "BaseSettingShootingMode=" + str;
        set_command<CGICmd::st_Project>(msg);
    }

    void set_project_RecFormatVideoFormat(CGICmd::em_RecFormatVideoFormat val)
    {
        std::string msg;
        auto str = json_conv_enum2str(val);
        msg = "RecFormatVideoFormat=" + str;
        set_command<CGICmd::st_Project>(msg);
    }



    /////////////////////////////////////////////////////////////////
    // stream.
    void set_stream_StreamMode(CGICmd::em_StreamMode mode)
    {
        auto str = json_conv_enum2str(mode);
        std::string msg = "StreamMode=" + str;
        set_command<CGICmd::st_Stream>(msg);
    }



    /////////////////////////////////////////////////////////////////
    // inquiry.
    template<typename T> void inquiry(T& val)
    {
        if (!auth) return;

        std::string msg = "/command/inquiry.cgi?inqjson=";
        msg += T::cmd;
        auto [s, s_msg, r] = GET(msg);

        if (s == 0 && s_msg == "" && r == "") return;

        switch (s) {
        case cpr::status::HTTP_UNAUTHORIZED:
        // case cpr::status::HTTP_REQUEST_TIMEOUT:
            std::cout << "ERROR (" << s << ") : " << s_msg << std::endl;
            auth = false;
            break;

        case cpr::status::HTTP_OK:
        // case SC::Created_201:
        // case SC::Accepted_202:
        // case SC::NonAuthoritativeInformation_203:
        // case SC::NoContent_204:
        // case SC::ResetContent_205:
        // case SC::PartialContent_206:
        // case SC::MultiStatus_207:
        // case SC::AlreadyReported_208:
        // case SC::IMUsed_226:
            try
            {
                auto j = njson::parse(r);
                auto j_sub = j[T::cmd];
                val = j_sub.template get<T>();
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }
            break;

        default:
            std::cout << "(" << s << ") : " << s_msg << std::endl;
        }
    }
    auto &inquiry_system() { return cmd_info.system; }
    auto &inquiry_status() { return cmd_info.status; }
    auto &inquiry_ptzf() { return cmd_info.ptzf; }
    auto &inquiry_imaging() { return cmd_info.imaging; }
    auto &inquiry_cameraoperation() { return cmd_info.cameraoperation; }
    auto &inquiry_tally() { return cmd_info.tally; }
    auto &inquiry_project() { return cmd_info.project; }
    auto &inquiry_network() { return cmd_info.network; }
    auto &inquiry_srt() { return cmd_info.srt; }



    bool is_update_cmd_info() { return is_update_cmd_info_list.load(); }

    void fetch(bool latest = false)
    {
        std::lock_guard<std::mutex> lg(mtx);

        cmd_info = latest ? cmd_info_list->PeekLatest() : cmd_info_list->Peek();
        is_update_cmd_info_list.store(false);
    }

    void next(bool latest = false)
    {
        std::lock_guard<std::mutex> lg(mtx);

        latest ? cmd_info_list->ReadLatest() : cmd_info_list->Read();
    }

    void run_inq()
    {
        TinyTimer tt;

        while (running)
        {
            std::tie(lap_cur_inq, lap_ave_inq) = sw_inq.lap();

            st_CmdInfo cmdi = {};

            inquiry(cmdi.system);
            if (timeout || !connected) continue;
            inquiry(cmdi.status);
            inquiry(cmdi.ptzf);
            inquiry(cmdi.imaging);
            inquiry(cmdi.cameraoperation);
            inquiry(cmdi.tally);
            inquiry(cmdi.project);
            inquiry(cmdi.network);
            inquiry(cmdi.stream);
            inquiry(cmdi.srt);

            if (auth) {
                if (!is_send_set_cmd.load())
                {
                    std::lock_guard<std::mutex> lg(mtx);
                    cmd_info_list->Write(cmdi);
                    is_update_cmd_info_list.store(true);
                } else {
                    is_send_set_cmd.store(false);
                }
            }

            tt.wait1period(1.0 / 60.0f);
        }
    }

    void run_set()
    {
        while (running)
        {
            std::tie(lap_cur_set, lap_ave_set) = sw_set.lap();

            const auto wait_time = std::chrono::microseconds(16667);
            std::unique_lock<std::mutex> lk(mtx_cmd_msg);
            auto ret = cv_cmd_msg.wait_for(lk, wait_time, [&]{ return is_update_cmd_msg_list.load(); });
            if (!ret) continue;

            auto msg = cmd_msg_list->Read();
            is_update_cmd_msg_list.store(false);
            if (!msg.empty()) {
                is_send_set_cmd.store(true);
                auto [s, s_msg, r] = GET(msg);

                if (s == cpr::status::HTTP_UNAUTHORIZED) {
                    auth = false;
                }

                if (s == cpr::status::HTTP_REQUEST_TIMEOUT) {
                    std::cerr << "[run_set] Time Out" << std::endl;
                }
            }
        }
    }
};
