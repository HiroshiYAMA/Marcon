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

}   // namespace COMMON.



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

    CGICmd::COMMON::em_TrueFalse TemperatureWarning;

public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        st_Status,

        TemperatureWarning
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



struct st_Range
{
    int min;
    int max;
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
};



/////////////////////////////////////////////////////////////////////
// project.
struct st_Project
{
    static constexpr auto cmd = "project";

    std::string RecFormatFrequency;

public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        st_Project,

        RecFormatFrequency
    )
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



void to_json(njson& j, const CGICmd::st_Imaging& p);
void from_json(const njson& j, CGICmd::st_Imaging& p);

void to_json(njson& j, const CGICmd::st_Network& p);
void from_json(const njson& j, CGICmd::st_Network& p);

void to_json(njson& j, const CGICmd::st_Srt& p);
void from_json(const njson& j, CGICmd::st_Srt& p);

}   // namespace CGICmd.



class CGI
{
private:
    // HTTP
    std::unique_ptr<httplib::Client> client;

    // command.
    bool auth;
    struct st_CmdInfo
    {
        CGICmd::st_System system;
        CGICmd::st_Status status;
        CGICmd::st_Imaging imaging;
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
    {
        client = std::make_unique<httplib::Client>(ip_address, port);

        if (client) {
            set_account(id, pw);
            set_referer_message(ip_address, port);
        }

        auth = false;

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

    bool is_auth() const { return auth; }

    bool empty_cmd_msg_list() const { return cmd_msg_list->empty(); }
    bool full_cmd_msg_list() const { return cmd_msg_list->full(); }

    std::tuple<double, double> get_lap_inq() { return { lap_cur_inq, lap_ave_inq }; }
    std::tuple<double, double> get_lap_set() { return { lap_cur_set, lap_ave_set }; }

    // auto &get_cmd_info() { return cmd_info; }
    // auto &get_cmd_info() const { return cmd_info; }

    void set_account(const std::string &id, const std::string &pw)
    {
        std::lock_guard<std::mutex> lg(mtx_auth);

        client->set_digest_auth(id, pw);
        auth = true;
    }

    void set_referer_message(const std::string &server_adr, int server_port)
    {
        auto referer_str = make_referer_message(server_adr, server_port);

        client->set_default_headers({
            { "Referer", referer_str }
        });
    }

    std::tuple<int, std::string> GET(const std::string &message)
    {
        std::lock_guard<std::mutex> lg(mtx_auth);

        if (!auth) return { 0, "" };

        auto res = client->Get(message);

        if (!res) return { 0, "" };

        auto status = res->status;
        auto body = res->body;

        if (status == httplib::StatusCode::Unauthorized_401) auth = false;

        return { status, body };
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
        std::string msg = "/command/inquiry.cgi?inqjson=";
        msg += T::cmd;
        auto [s, r] = GET(msg);

        if (s == 0 && r == "") return;

        auto s_msg = httplib::status_message(s);
        switch (s) {
            using SC = httplib::StatusCode;

        case SC::Unauthorized_401:
            std::cout << "ERROR (" << s << ") : " << s_msg << std::endl;
            throw SC::Unauthorized_401;
            break;

        case SC::OK_200:
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
    auto &inquiry_imaging() { return cmd_info.imaging; }
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

            try
            {
                inquiry(cmdi.system);
                inquiry(cmdi.status);
                inquiry(cmdi.imaging);
                inquiry(cmdi.project);
                inquiry(cmdi.network);
                inquiry(cmdi.stream);
                inquiry(cmdi.srt);
            }
            catch(const httplib::StatusCode &s)
            {
                if (s == httplib::StatusCode::Unauthorized_401) {
                    auth = false;
                }
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }

            if (!is_send_set_cmd.load())
            {
                std::lock_guard<std::mutex> lg(mtx);
                cmd_info_list->Write(cmdi);
                is_update_cmd_info_list.store(true);
            } else {
                is_send_set_cmd.store(false);
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
                try
                {
                    is_send_set_cmd.store(true);
                    auto [s, r] = GET(msg);
                }
                catch(const httplib::StatusCode &s)
                {
                    if (s == httplib::StatusCode::Unauthorized_401) {
                        auth = false;
                    }
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }
            }
        }
    }
};
