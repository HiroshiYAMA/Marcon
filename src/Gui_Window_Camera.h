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

#include <atomic>
#include <optional>

#include "common_utils.h"
#include "gui_utils.h"
#include "RemoteServer.h"
#include "ProcLiveView.h"
#include "CGI.h"
#include "IpNetwork.h"

#include "Gui_Window_Keyboard.h"

#if __has_include(<charconv>)
#include <charconv>
#endif

inline auto make_combo_vec = [](auto &vec, auto &combo_vec, auto &len_max) -> void {
    for (auto &[k, v] : vec) {
        len_max = std::max(len_max, v.length() * 8);
        for (auto i = 0; i < v.size(); i++) {
            combo_vec.push_back(v[i]);
        }
        combo_vec.push_back('\0');
    }
    combo_vec.push_back('\0');
};

inline auto make_listbox_vec = [](auto &vec, auto &listbox_vec, auto &len_max) -> void {
    for (auto &[k, v] : vec) {
        len_max = std::max(len_max, v.length() * 8);
        listbox_vec.push_back(v.c_str());
    }
};

enum class em_Camera_Connection_State : int {
    DISCONNECTED,
    NO_AUTH,
    CONNECTED
};

class Gui_Window_Camera
{
public:
    enum class em_State {
        MAIN,
        SHUTTER,
        WHITE_BALANCE,
        ISO,
        IRIS,
        ND,
        SYSTEM,

        STREAMING,

        LIVE_VIEW,
    };

    enum class em_System_State {
        MAIN,
        FPS,
        VIDEO_FORMAT,
        SHOOTING_MODE,
    };

    enum class em_Ptzf_State {
        PTZ,
        FOCUS,
    };

    enum class em_StateShutter {
        MODE,
        CONTROL,
    };

    enum class em_StateWhiteBalance {
        MODE,
        CONTROL,
    };

    enum class em_StateISO {
        MODE_AUTO_MANUAL,
        MODE_BASE_SENSITIVITY,
        MODE_BASE_ISO = MODE_BASE_SENSITIVITY,
        CONTROL,
    };

    enum class em_StateIris {
        MODE,
        CONTROL,
    };

    enum class em_StateND {
        MODE,
        CONTROL,
    };

private:
    const std::list<std::pair<CGICmd::em_ExposureShutterModeState, std::string>> exposure_shutter_mode_state = {
        {CGICmd::ExposureShutterModeState_AUTO, "Auto"},
        {CGICmd::ExposureShutterModeState_SPEED, "Speed"},
        {CGICmd::ExposureShutterModeState_ANGLE, "Angle"},
        {CGICmd::ExposureShutterModeState_ECS, "ECS"},
        {CGICmd::ExposureShutterModeState_OFF, "Off"},
    };

    const std::list<std::pair<CGICmd::em_WhiteBalanceModeState, std::string>> white_balance_mode_state = {
        {CGICmd::em_WhiteBalanceModeState::AUTO, "ATW"},
        {CGICmd::em_WhiteBalanceModeState::MANUAL, "Manual"},
    };

    const std::list<std::pair<CGICmd::em_ISOModeState, std::string>> iso_mode_state = {
        {CGICmd::em_ISOModeState::GAIN, "Gain"},
        {CGICmd::em_ISOModeState::ISO, "ISO"},
        {CGICmd::em_ISOModeState::CINE_EI_QUITCK, "Cine EI Quick"},
        {CGICmd::em_ISOModeState::CINE_EI, "Cine EI"},
    };

    const std::list<std::pair<CGICmd::COMMON::em_OnOff, std::string>> iso_agc_on_off = {
        {CGICmd::COMMON::ON, "AGC"},
        {CGICmd::COMMON::OFF, "Manual"},
    };

    const std::list<std::pair<CGICmd::COMMON::em_LowHigh, std::string>> iso_base_sensitivity = {
        {CGICmd::COMMON::LOW, "Base Low"},
        {CGICmd::COMMON::HIGH, "Base High"},
    };

    const std::list<std::pair<CGICmd::em_ExposureBaseISO, std::string>> iso_base_iso = {
        {CGICmd::ExposureBaseISO_ISO800, "ISO 800"},
        {CGICmd::ExposureBaseISO_ISO12800, "ISO 12800"},
    };

    const std::list<std::pair<CGICmd::em_IrisModeState, std::string>> iris_mode_state = {
        {CGICmd::em_IrisModeState::AUTO, "Auto"},
        {CGICmd::em_IrisModeState::MANUAL, "Manual"},
    };

    const std::list<std::pair<CGICmd::COMMON::em_OnOff, std::string>> iris_auto_on_off = {
        {CGICmd::COMMON::ON, "Auto"},
        {CGICmd::COMMON::OFF, "Manual"},
    };

    const std::list<std::pair<CGICmd::em_NDModeState, std::string>> nd_mode_state = {
        {CGICmd::em_NDModeState::AUTO, "Auto"},
        {CGICmd::em_NDModeState::MANUAL, "Manual"},
        {CGICmd::em_NDModeState::CLEAR, "Clear"},
    };

    const std::list<std::pair<CGICmd::COMMON::em_OnOff, std::string>> nd_auto_on_off = {
        {CGICmd::COMMON::ON, "Auto"},
        {CGICmd::COMMON::OFF, "Manual"},
    };

    const std::list<std::pair<CGICmd::em_ExposureNDClear, std::string>> nd_clear_on_off = {
        {CGICmd::ExposureNDClear_CLEAR, "Clear"},
        {CGICmd::ExposureNDClear_FILTERED, "Filtered"},
    };

    const std::list<std::pair<CGICmd::em_RecFormatFrequency, std::string>> system_fps = {
        {CGICmd::RecFormatFrequency_5994, "59.94p"},
        {CGICmd::RecFormatFrequency_5000, "50p"},
        {CGICmd::RecFormatFrequency_2997, "29.97p"},
        {CGICmd::RecFormatFrequency_2500, "25p"},
        {CGICmd::RecFormatFrequency_2400, "24p"},
        {CGICmd::RecFormatFrequency_2398, "23.98p"},
    };

    const std::list<std::pair<CGICmd::em_RecFormatVideoFormat, std::string>> system_video_format = {
        {CGICmd::RecFormatVideoFormat_4096x2160p, "4096x2160p"},
        {CGICmd::RecFormatVideoFormat_3840x2160p, "3840x2160p"},
        {CGICmd::RecFormatVideoFormat_1920x1080p, "1920x1080p"},
        {CGICmd::RecFormatVideoFormat_1920x1080p_50, "1920x1080p_50"},
        {CGICmd::RecFormatVideoFormat_1920x1080p_35, "1920x1080p_35"},
    };

    const std::list<std::pair<CGICmd::em_BaseSettingShootingMode, std::string>> system_shooting_mode = {
        {CGICmd::BaseSettingShootingMode_CUSTOM, "Custom"},
        {CGICmd::BaseSettingShootingMode_FLEXIBLE_ISO, "Flexible ISO"},
        {CGICmd::BaseSettingShootingMode_CINE_EI, "Cine EI"},
        {CGICmd::BaseSettingShootingMode_CINE_EI_QUICK, "Cine EI Quick"},
    };

    st_RemoteServer remote_server = {};
    Gui_Window_Keyboard kb = {};
    bool display_input_username = false;
    bool display_input_password = false;

    CGICmd::em_StreamMode stream_mode_bkup;

    em_State stat_main = em_State::MAIN;
    em_State stat_main_bkup = em_State::MAIN;

    em_System_State stat_system = em_System_State::MAIN;
    CGICmd::em_RecFormatFrequency rec_fps_tmp;

    em_Ptzf_State stat_ptzf = em_Ptzf_State::PTZ;

    // shutter.
    em_StateShutter stat_shutter = em_StateShutter::CONTROL;
    CGICmd::em_ExposureShutterModeState shutter_mode_state_bkup = CGICmd::ExposureShutterModeState_AUTO;

    // white balance.
    em_StateWhiteBalance stat_wb = em_StateWhiteBalance::CONTROL;
    CGICmd::em_WhiteBalanceModeState wb_mode_state_bkup = CGICmd::em_WhiteBalanceModeState::AUTO;

    // ISO.
    em_StateISO stat_iso = em_StateISO::CONTROL;
    CGICmd::em_ISOModeState iso_mode_state_bkup = CGICmd::em_ISOModeState::GAIN;

    // IRIS.
    em_StateIris stat_iris = em_StateIris::CONTROL;
    CGICmd::em_IrisModeState iris_mode_state_bkup = CGICmd::em_IrisModeState::AUTO;
    using iris_list_t = CGICmd::shutter_list;
    iris_list_t exposure_iris;

    // ND.
    em_StateND stat_nd = em_StateND::CONTROL;
    CGICmd::em_NDModeState nd_mode_state_bkup = CGICmd::em_NDModeState::AUTO;

    // cudaStream_t m_cuda_stream = NULL;

    GLsizei tex_width;
    GLsizei tex_height;
    GLenum tex_type;
    GLint tex_internalFormat;
    GLenum tex_format;
    GLuint tex_id;

    std::atomic<em_Camera_Connection_State> camera_connection_stat = em_Camera_Connection_State::DISCONNECTED;

    // thread.
    std::unique_ptr<ProcLiveView> proc_live_view;
    std::thread thd_proc_live_view;
    // bool is_display_image;
    int queue_size;

    // CGI.
    std::unique_ptr<CGI> cgi;
    std::thread thd_cgi_inq;
    std::thread thd_cgi_set;

    // VISCA.
    std::unique_ptr<IpNetwork::VISCA_Com> visca_com;

    StopWatch sw;

    // blinking tally.
    std::thread thd_blink_tally;
    std::atomic_bool fin_thd_blink_tally;
    float hue_blink_tally_button = 0.0f;
#if 0
    void blink_tally(bool green, bool red)
    {
        auto tally_bkup = cgi->inquiry_tally();

        cgi->set_tally_GTallyLampEnable(CGICmd::COMMON::em_OnOff::ON);
        cgi->set_tally_TallyControlMode(CGICmd::em_InternalExternal::EXTERNAL);
        cgi->set_tally_TallyLampBrightness(CGICmd::em_OffLowHigh::HIGH);

        bool is_on = true;
        for (auto i = 0; i < 20; i++) {
            // Send Tally command.
            auto on_off = is_on ? CGICmd::em_TurnOnOff::TURN_ON : CGICmd::em_TurnOnOff::TURN_OFF;
            if (green) cgi->set_tally_GTallyControl(on_off);
            if (red) cgi->set_tally_RTallyControl(on_off);
            is_on = !is_on;

            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }

        // restore.
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        cgi->set_tally_GTallyLampEnable(tally_bkup.GTallyLampEnable);
        cgi->set_tally_TallyControlMode(tally_bkup.TallyControlMode);
        cgi->set_tally_TallyLampBrightness(tally_bkup.TallyLampBrightness);
        cgi->set_tally_GTallyControl(tally_bkup.GTallyControl);
        cgi->set_tally_RTallyControl(tally_bkup.RTallyControl);

        fin_thd_blink_tally.store(true);
    }
#else
    using tally_color_t = IpNetwork::VISCA_Tally_Command::em_COLOR;
    void blink_tally(const std::string &name, tally_color_t color = tally_color_t::RED)
    {
        // backup & set.
        auto tally_bkup = cgi->inquiry_tally();
        cgi->set_comb_tally(CGICmd::COMMON::em_OnOff::ON, CGICmd::em_InternalExternal::EXTERNAL, CGICmd::em_OffLowHigh::HIGH);

        bool is_on = true;
        for (auto i = 0; i < 20; i++) {
            // Send Tally command.
            auto color = tally_color_t::RED;
            visca_com->send_cmd_tally(color, is_on);

            is_on = !is_on;

            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }

        // restore.
        cgi->set_comb_tally(tally_bkup.GTallyLampEnable, tally_bkup.TallyControlMode, tally_bkup.TallyLampBrightness);

        fin_thd_blink_tally.store(true);
    }
#endif

#if 0
    void show_panel_blink_tally(bool green, bool red)
    {
        ImGuiStyle& style = ImGui::GetStyle();
        const auto text_size = ImGui::CalcTextSize("A");
        const auto pad_frame = style.FramePadding;
        const ImVec2 btn_size(text_size.x + pad_frame.x * 2, (text_size.y + pad_frame.y) * 2);

        auto is_set_color = false;
        constexpr auto hue_max = 8.0f;
        auto idx_btn_text = static_cast<int>(hue_blink_tally_button * 4) % 4;
        auto btn_text = fin_thd_blink_tally.load() ? "@T@##Tally"
            : idx_btn_text == 0 ? " | ##Tally"
            : idx_btn_text == 1 ? " / ##Tally"
            : idx_btn_text == 2 ? " - ##Tally"
            : " \\ ##tally"
            ;
        if (!fin_thd_blink_tally.load()) {
            set_style_color(hue_blink_tally_button / hue_max, 0.7f, 0.7f);
            hue_blink_tally_button += 0.1f;
            if (hue_blink_tally_button > hue_max) hue_blink_tally_button = 0.0f;
            is_set_color = true;
        }
        if (ImGui::Button(btn_text, ImVec2(0, btn_size.y))) {
            if (!thd_blink_tally.joinable()) {
                fin_thd_blink_tally.store(false);
                std::thread thd_tmp{ [&]{ blink_tally(green, red); }};
                thd_blink_tally = std::move(thd_tmp);
            }
        }
        if (thd_blink_tally.joinable() && fin_thd_blink_tally.load()) thd_blink_tally.join();
        if (is_set_color) reset_style_color();
    }
#else
    void show_panel_blink_tally(const std::string &ip_address)
    {
        ImGuiStyle& style = ImGui::GetStyle();
        const auto text_size = ImGui::CalcTextSize("A");
        const auto pad_frame = style.FramePadding;
        const ImVec2 btn_size(text_size.x + pad_frame.x * 2, (text_size.y + pad_frame.y) * 2);

        auto is_set_color = false;
        constexpr auto hue_max = 8.0f;
        auto idx_btn_text = static_cast<int>(hue_blink_tally_button * 4) % 4;
        auto btn_text = fin_thd_blink_tally.load() ? "@T@##Tally"
            : idx_btn_text == 0 ? " | ##Tally"
            : idx_btn_text == 1 ? " / ##Tally"
            : idx_btn_text == 2 ? " - ##Tally"
            : " \\ ##tally"
            ;
        if (!fin_thd_blink_tally.load()) {
            set_style_color(hue_blink_tally_button / hue_max, 0.7f, 0.7f);
            hue_blink_tally_button += 0.1f;
            if (hue_blink_tally_button > hue_max) hue_blink_tally_button = 0.0f;
            is_set_color = true;
        }
        if (ImGui::Button(btn_text, ImVec2(0, btn_size.y))) {
            if (!thd_blink_tally.joinable()) {
                fin_thd_blink_tally.store(false);
                std::thread thd_tmp{ [&]{ blink_tally(ip_address); }};
                thd_blink_tally = std::move(thd_tmp);
            }
        }
        if (thd_blink_tally.joinable() && fin_thd_blink_tally.load()) thd_blink_tally.join();
        if (is_set_color) reset_style_color();
    }
#endif

    // print selected camera info.
    void show_panel_print_selected_camera_info()
    {
        ImGui::PushID("Print_Selected_Camera_Info");

        auto &rs = remote_server;
        std::string str = rs.ip_address + ":" + rs.port + " / " + "[SRT]" + (rs.is_srt_listener ? "Listener" : "Caller") + ":" + rs.srt_port;
        ImGui::Text("%s", str.c_str());

        ImGui::PopID();
    }

    // print FPS.
    void show_panel_print_fps()
    {
        ImGui::PushID("Print_Fps");

        {
            auto [lap_cur, lap_ave] = proc_live_view->get_lap();
            ImGui::Text("LiveView %.2f(ms) / %.2f(fps)", lap_ave, 1'000.0 / lap_ave);
        }

        {
            auto [lap_cur, lap_ave] = cgi->get_lap_inq();
            ImGui::Text("CGI(inq) %.2f(ms) / %.2f(fps)", lap_ave, 1'000.0 / lap_ave);
        }
        {
            auto [lap_cur, lap_ave] = cgi->get_lap_set();
            ImGui::Text("CGI(set) %.2f(ms) / %.2f(fps)", lap_ave, 1'000.0 / lap_ave);
        }

        ImGui::PopID();
    }

    // movie rec.
    void show_panel_movie_rec()
    {
        ImGui::PushID("Movie_Rec");

        set_style_color(0.0f / 7.0f);

        std::string str = "REC";
        if (ImGui::Button(str.c_str()) || ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
            cgi->click_cameraoperation_MediaRecording();
        }

        ImGui::SameLine();

        auto &cameraoperation = cgi->inquiry_cameraoperation();
        auto is_rec = (cameraoperation.MediaRecordingStatus == CGICmd::em_MediaRecordingStatus::MediaRecordingStatus_REC);
        ImVec4 color;
        std::string status_str;
        if (is_rec) {
            color = (gui_skin != em_GuiSkin::LIGHT) ? ImVec4{1, 0, 0, 1} : ImVec4{0.5f, 0, 0, 1};
            status_str = "Recording";
        } else {
            color = (gui_skin != em_GuiSkin::LIGHT) ? ImVec4{0, 1, 0, 1} : ImVec4{0, 0.5f, 0, 1};
            status_str = "Not Recording";
        }
        ImGui::TextColored(color, "%s", status_str.c_str());

        reset_style_color();

        ImGui::PopID();
    }

    void centering_text_pos(const char *text)
    {
        static std::atomic<uint64_t> id(0);

        std::string id_str = "centering_text_pos";
        auto id_tmp = id.load();
        id_str += std::to_string(id_tmp);
        id.store(id_tmp++);

        ImVec2 p_center;
        if (ImGui::BeginChild(id_str.c_str(), ImVec2(-1, -1), ImGuiChildFlags_None, ImGuiWindowFlags_None))
        {
                auto p = ImGui::GetCursorScreenPos();
                auto sz = ImGui::GetWindowSize();
                auto sz_text = ImGui::CalcTextSize(text);
                p_center = ImVec2(p.x + (sz.x / 2) - (sz_text.x / 2), p.y);
        }
        ImGui::EndChild();
        ImGui::SetCursorScreenPos(p_center);
    }

#if 0   // TODO: something's wrong.
    void centering_text_pos_v(int row_offset = 1)
    {
        static std::atomic<uint64_t> id(0);

        std::string id_str = "centering_text_pos_v";
        auto id_tmp = id.load();
        id_str += std::to_string(id_tmp);
        id.store(id_tmp++);

        ImVec2 p_center;
        if (ImGui::BeginChild(id_str.c_str(), ImVec2(-1, -1), ImGuiChildFlags_None, ImGuiWindowFlags_None))
        {
            auto p = ImGui::GetCursorScreenPos();
            auto sz = ImGui::GetWindowSize();
            auto text_height = ImGui::GetTextLineHeightWithSpacing();
            auto p_center = ImVec2(p.x, p.y + sz.y / 2 - text_height * row_offset);
        }
        ImGui::EndChild();
        ImGui::SetCursorScreenPos(p_center);
    }
#endif

    template<typename S, typename V> void show_panel_state_mode_str(S state, const V &vec, bool center = false)
    {
        ImGui::PushID("STATE_MODE_STR");

        std::string mode_str = get_string_from_pair_list(vec, state);

        if (center) {
            centering_text_pos(mode_str.c_str());
        }

        ImGui::Text("%s", mode_str.c_str());

        ImGui::PopID();
    }

    // select value listbox.
    template<typename T, typename Tfunc=std::function<void(T)>> void show_panel_select_value_listbox(
        const char *id_str,
        T &idx, T idx_min, T idx_max,
        const std::list<std::pair<T, std::string>> &lst,
        Tfunc func_set,
        float key_delay = 0.275f, float key_rate = 0.050f
    )
    {
        idx = static_cast<T>(std::clamp(static_cast<int>(idx), static_cast<int>(idx_min), static_cast<int>(idx_max)));

        auto &vec = lst;
        {
            ImGui::PushID(id_str);

            ImGuiStyle& style = ImGui::GetStyle();

            ImGui::BeginGroup();

            if (ImGui::BeginChild("CHILD", ImVec2(-1, -1), ImGuiChildFlags_None, ImGuiWindowFlags_None))
            {
                ImVec2 p = ImGui::GetCursorScreenPos();
                ImVec2 win_size = ImGui::GetWindowSize();

                auto &io = ImGui::GetIO();
                auto key_delay_bkup = io.KeyRepeatDelay;
                auto key_rate_bkup = io.KeyRepeatRate;
                io.KeyRepeatDelay = key_delay;
                io.KeyRepeatRate = key_rate;

                auto itr = std::find_if(vec.begin(), vec.end(), [&idx](auto &e){ return e.first == idx; });
                bool is_changed = false;

                // centering.
                constexpr auto btn_scale = 2.0f;
                ImGui::SetWindowFontScale(btn_scale);
                const auto text_size = ImGui::CalcTextSize("A");
                const auto pad_frame = style.FramePadding;
                const ImVec2 ar_size(text_size.x + pad_frame.x, text_size.y + pad_frame.y);
                ImVec2 p_up(p.x + (win_size.x / 2) - (ar_size.x / 2) - pad_frame.x * btn_scale, p.y + 4.0f);
                ImGui::SetCursorScreenPos(p_up);
                ImGui::PushButtonRepeat(true);
                if (ImGui::ArrowButton("##UP", ImGuiDir_Up)
                    || ImGui::IsKeyPressed(ImGuiKey_E)
                    || ImGui::IsKeyPressed(ImGuiKey_UpArrow)
                ) {
                    if (itr != vec.begin()) itr--;
                    is_changed = true;
                }
                ImGui::PopButtonRepeat();
                ImGui::SetWindowFontScale(1.0f);

                ImVec2 p_list(p.x, p.y + (win_size.y * (1.0f - 0.5f) / 2));
                ImGui::SetCursorScreenPos(p_list);
                if (ImGui::BeginChild("CHILD_CHILD", ImVec2(-1, win_size.y * 0.5f), ImGuiChildFlags_Border, ImGuiWindowFlags_None))
                {
                    ImGuiStyle& style = ImGui::GetStyle();
                    auto fr = style.FrameRounding;
                    auto gr = style.GrabRounding;

                    style.FrameRounding = 0.0f;
                    style.GrabRounding = 0.0f;

                    auto fb = style.FrameBorderSize;

                    ImGui::SetWindowFontScale(1.5f);
                    for (auto &[k, v]: vec) {
                        if (k == idx) {
                            set_style_color(4.0f / 7.0f, 0.9f, 0.9f);

                            ImGui::Button(v.c_str(), ImVec2(-1, 0));
                            ImGui::SetScrollHereY(0.5f); // 0.0f:top, 0.5f:center, 1.0f:bottom

                            reset_style_color();

                        } else {
                            style.FrameBorderSize = 0.0f;
                            set_style_color(5.0f, 0.1f, 0.1f, 0.0f);

                            ImGui::Button(v.c_str(), ImVec2(-1, 0));

                            reset_style_color();
                            style.FrameBorderSize = fb;
                        }
                    }
                    ImGui::SetWindowFontScale(1.0f);

                    style.FrameRounding = fr;
                    style.GrabRounding = gr;
                }
                ImGui::EndChild();

                // centering.
                ImGui::SetWindowFontScale(btn_scale);
                ImVec2 p_down(p_up.x, win_size.y - ar_size.y + (0.0f - 4.0f));
                ImGui::SetCursorScreenPos(p_down);
                ImGui::PushButtonRepeat(true);
                if (ImGui::ArrowButton("##DOWN", ImGuiDir_Down)
                    || ImGui::IsKeyPressed(ImGuiKey_C)
                    || ImGui::IsKeyPressed(ImGuiKey_DownArrow)
                ) {
                    itr++;
                    if (itr == vec.end()) itr--;
                    is_changed = true;
                }
                ImGui::PopButtonRepeat();
                ImGui::SetWindowFontScale(1.0f);

                io.KeyRepeatDelay = key_delay_bkup;
                io.KeyRepeatRate = key_rate_bkup;

                if (is_changed) {
                    auto &[k, v] = *itr;
                    idx = k;   // pre-set for GUI.
                    func_set(k);
                }

            }
            ImGui::EndChild();

            ImGui::EndGroup();

            ImGui::PopID();
        }
    }

    void show_panel_text_select_mode()
    {
        if (ImGui::BeginChild("CHILD text_select_mode", ImVec2(-1, -1), ImGuiChildFlags_None, ImGuiWindowFlags_None))
        {
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImVec2 win_size = ImGui::GetWindowSize();

            auto &io = ImGui::GetIO();
            auto &style = ImGui::GetStyle();

            // centering.
            constexpr auto str = "Select Mode -->";
            const auto text_size = ImGui::CalcTextSize(str);
            const auto pad_frame = style.FramePadding;
            const ImVec2 ar_size(text_size.x + pad_frame.x, text_size.y + pad_frame.y);
            ImVec2 p_up(p.x + (win_size.x / 2) - (ar_size.x / 2) - pad_frame.x, p.y + 4.0f);

            ImVec2 p_down(p_up.x, win_size.y - ar_size.y + (0.0f - 4.0f));
            ImGui::SetCursorScreenPos(p_down);
            auto col = (ImVec4)ImColor::HSV(0.75f / 7.0f, 1.0f, 1.0f);
            ImGui::TextColored(col, "%s", str);
        }
        ImGui::EndChild();
    }

    void show_panel_text_select_mode_right_down()
    {
        if (ImGui::BeginChild("CHILD text_select_mode_right_down", ImVec2(-1, -1), ImGuiChildFlags_None, ImGuiWindowFlags_None))
        {
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImVec2 win_size = ImGui::GetWindowSize();

            auto &io = ImGui::GetIO();
            auto &style = ImGui::GetStyle();

            // centering.
            constexpr auto str = "Select Mode --\\";
            const auto text_size = ImGui::CalcTextSize(str);
            const auto pad_frame = style.FramePadding;
            const ImVec2 ar_size(text_size.x + pad_frame.x, text_size.y + pad_frame.y);
            ImVec2 p_up(p.x + (win_size.x / 2) - (ar_size.x / 2) - pad_frame.x, p.y + 4.0f);
            ImGui::SetCursorScreenPos(p_up);
            auto col = (ImVec4)ImColor::HSV(0.75f / 7.0f, 1.0f, 1.0f);
            ImGui::TextColored(col, "%s", str);
        }
        ImGui::EndChild();
    }

    void show_panel_text_select_mode_right_up_down()
    {
        if (ImGui::BeginChild("CHILD text_select_mode_right_up_down", ImVec2(-1, -1), ImGuiChildFlags_None, ImGuiWindowFlags_None))
        {
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImVec2 win_size = ImGui::GetWindowSize();

            auto &io = ImGui::GetIO();
            auto &style = ImGui::GetStyle();
            const auto pad_frame = style.FramePadding;

            // centering.
            auto put_text = [&](const char *str, bool is_top) -> void {
                const auto text_size = ImGui::CalcTextSize(str);
                const ImVec2 ar_size(text_size.x + pad_frame.x, text_size.y + pad_frame.y);
                ImVec2 p_top(p.x + (win_size.x / 2) - (ar_size.x / 2) - pad_frame.x, p.y + 4.0f);
                ImVec2 p_btm(p.x + (win_size.x / 2) - (ar_size.x / 2) - pad_frame.x, win_size.y - ar_size.y + (0.0f - 4.0f));
                auto p_center = is_top ? p_top : p_btm;
                ImGui::SetCursorScreenPos(p_center);
                auto col = (ImVec4)ImColor::HSV(0.75f / 7.0f, 1.0f, 1.0f);
                ImGui::TextColored(col, "%s", str);
            };

            put_text("Select Mode --\\", true);

            put_text("Select Mode --/", false);
        }
        ImGui::EndChild();
    }



    /////////////////////////////////////////////////////////////////
    // Main control panel.
    /////////////////////////////////////////////////////////////////
    bool show_panel_main()
    {
        ImGui::PushID("Main");

        bool ret = true;

        auto tbl_flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        auto cld_flags = ImGuiChildFlags_None;
        auto win_flags = ImGuiWindowFlags_None;

        auto &imaging = cgi->inquiry_imaging();

        auto &style = ImGui::GetStyle();
        auto frm_padding = style.FramePadding;
        auto win_size = ImGui::GetWindowSize();

        auto btn_width = win_size.x / 3 - frm_padding.x * 4;
        auto btn_size = ImVec2(btn_width, 0);

        auto left_x = win_size.x / 6 - btn_width / 2;
        auto center_x = left_x + win_size.x / 3;
        auto right_x = center_x + win_size.x / 3;

        constexpr auto font_scale = 1.5f;
        auto spacing = []() { ImGui::Text(" "); ImGui::SameLine(); };

        if (ImGui::BeginTable("main", 1, tbl_flags))
        {
            auto p = ImGui::GetCursorScreenPos();
            auto sz = ImGui::GetWindowSize();
            float min_row_height = (sz.y - p.y - 24) / 3;

            for (int row = 0; row < 3; row++)
            {
                ImGui::TableNextRow(ImGuiTableRowFlags_None, min_row_height);
                ImGui::TableNextColumn();

                if (row == 0) {
                    ImGui::SetWindowFontScale(font_scale);
                    if (ImGui::BeginTable("main_top", 3, ImGuiTableFlags_BordersInnerV))
                    {
                        for (int row = 0; row < 1; row++)
                        {
                            ImGui::TableNextRow();

                            ///////////////////////////////////////////////////////////////////////
                            // TOP LEFT.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(0);
                            {
                                if (ImGui::BeginChild("main_top_left", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    auto p = ImGui::GetCursorScreenPos();

                                    ImGui::SetCursorScreenPos(ImVec2(left_x, p.y));
                                    auto is_press = (ImGui::Button("SYSTEM", btn_size) || ImGui::IsKeyPressed(ImGuiKey_W, false));

                                    spacing(); show_panel_system_fps_str();
                                    spacing(); show_panel_system_video_format_str();

                                    ImGui::SetCursorScreenPos(p);
                                    is_press |= ImGui::InvisibleButton("##SYSTEM", ImVec2(-1, -1), ImGuiButtonFlags_MouseButtonLeft);

                                    if (is_press) {
                                        stat_main = em_State::SYSTEM;
                                    }
                                }
                                ImGui::EndChild();
                            }

                            ///////////////////////////////////////////////////////////////////////
                            // TOP CENTER.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(1);
                            {
                                if (ImGui::BeginChild("main_top_center", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    auto p = ImGui::GetCursorScreenPos();

                                    auto idx = cgi->get_iso_mode_state();
                                    auto &vec = iso_mode_state;
                                    std::string mode_str = get_string_from_pair_list(vec, idx);

                                    ImGui::SetCursorScreenPos(ImVec2(center_x, p.y));
                                    auto is_press = (ImGui::Button(mode_str.c_str(), btn_size) || ImGui::IsKeyPressed(ImGuiKey_E, false));

                                    spacing();
                                    auto state = cgi->get_iso_mode_state();
                                    if (state == CGICmd::em_ISOModeState::GAIN) {
                                        show_panel_iso_base_sensitivity_str();
                                    } else if (state != CGICmd::em_ISOModeState::INVALID) {
                                        show_panel_iso_base_iso_str();
                                    }

                                    spacing();
                                    auto &imaging = cgi->inquiry_imaging();
                                    auto is_agc = imaging.ExposureAGCEnable == CGICmd::COMMON::ON;
                                    if (is_agc) {
                                        show_panel_iso_agc_str(); ImGui::SameLine();
                                        ImGui::Text(":"); ImGui::SameLine();
                                    }

                                    show_panel_iso_value();

                                    ImGui::SetCursorScreenPos(p);
                                    is_press |= ImGui::InvisibleButton("##ISO", ImVec2(-1, -1), ImGuiButtonFlags_MouseButtonLeft);

                                    if (is_press) {
                                        stat_main = em_State::ISO;
                                    }
                                }
                                ImGui::EndChild();
                            }

                            ///////////////////////////////////////////////////////////////////////
                            // TOP RIGHT.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(2);
                            {
                                if (ImGui::BeginChild("main_top_right", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    auto p = ImGui::GetCursorScreenPos();

                                    ImGui::SetCursorScreenPos(ImVec2(right_x, p.y));
                                    auto is_press = (ImGui::Button("Shutter", btn_size) || ImGui::IsKeyPressed(ImGuiKey_R, false));

                                    spacing(); show_panel_shutter_mode_str();
                                    spacing(); show_panel_shutter_value();

                                    ImGui::SetCursorScreenPos(p);
                                    is_press |= ImGui::InvisibleButton("##Shutter", ImVec2(-1, -1), ImGuiButtonFlags_MouseButtonLeft);

                                    if (is_press) {
                                        stat_main = em_State::SHUTTER;
                                    }
                                }
                                ImGui::EndChild();
                            }
                        }
                        ImGui::EndTable();
                    }
                    ImGui::SetWindowFontScale(1.0f);

                } else if (row == 1) {
                    if (ImGui::BeginTable("main_middle", 2, ImGuiTableFlags_BordersInnerV))
                    {
                        for (int row = 0; row < 1; row++)
                        {
                            ImGui::TableNextRow();

                            ///////////////////////////////////////////////////////////////////////
                            // MIDDLE LEFT.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(0);
                            {
                                ImGui::SetWindowFontScale(0.75f);

                                // print selected camera info.
                                show_panel_print_selected_camera_info();

                                // print FPS.
                                show_panel_print_fps();

                                ImGui::SetWindowFontScale(1.0f);

                                // movie rec.
                                ImGui::SetWindowFontScale(font_scale);
                                show_panel_movie_rec();
                                ImGui::SetWindowFontScale(1.0f);
                            }

                            ///////////////////////////////////////////////////////////////////////
                            // MIDDLE RIGHT.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(1);
                            {
                                ImVec2 p = ImGui::GetCursorScreenPos();
                                ImVec2 win_size = ImGui::GetWindowSize();

                                show_panel_live_view(win_size.x / 4, false, true);

                                ImGui::SetCursorScreenPos(p);
                                ImGui::InvisibleButton("##INVISIBULE_BUTTON", ImVec2(win_size.x / 4, min_row_height));
                                auto is_hovered = ImGui::IsItemHovered();
                                if (is_hovered && ImGui::IsMouseClicked(0)) {
                                    stat_main_bkup = stat_main;
                                    stat_main = em_State::LIVE_VIEW;
                                }

                                ImGui::SameLine();

                                // show_panel_blink_tally(false, true);
                                show_panel_blink_tally(remote_server.ip_address);
                            }
                        }
                        ImGui::EndTable();
                    }

                } else if (row == 2) {
                    ImGui::SetWindowFontScale(font_scale);
                    if (ImGui::BeginTable("main_bottom", 3, ImGuiTableFlags_BordersInnerV))
                    {
                        for (int row = 0; row < 1; row++)
                        {
                            ImGui::TableNextRow();

                            ///////////////////////////////////////////////////////////////////////
                            // BOTTOM LEFT.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(0);
                            {
                                if (ImGui::BeginChild("main_bottom_left", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    auto p = ImGui::GetCursorScreenPos();

                                    ImGui::SetCursorScreenPos(ImVec2(left_x, p.y));
                                    auto is_press = (ImGui::Button("ND", btn_size) || ImGui::IsKeyPressed(ImGuiKey_X, false));

                                    spacing(); show_panel_nd_mode_str();
                                    auto state = cgi->get_nd_mode_state();
                                    if (state == CGICmd::em_NDModeState::AUTO || state == CGICmd::em_NDModeState::MANUAL) {
                                        spacing(); show_panel_nd_value();
                                    }

                                    ImGui::SetCursorScreenPos(p);
                                    is_press |= ImGui::InvisibleButton("##ND", ImVec2(-1, -1), ImGuiButtonFlags_MouseButtonLeft);

                                    if (is_press) {
                                        stat_main = em_State::ND;
                                    }
                                }
                                ImGui::EndChild();
                            }

                            ///////////////////////////////////////////////////////////////////////
                            // BOTTOM CENTER.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(1);
                            {
                                if (ImGui::BeginChild("main_bottom_center", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    auto p = ImGui::GetCursorScreenPos();

                                    ImGui::SetCursorScreenPos(ImVec2(center_x, p.y));
                                    auto is_press = (ImGui::Button("IRIS", btn_size) || ImGui::IsKeyPressed(ImGuiKey_C, false));

                                    spacing(); show_panel_iris_mode_str();
                                    spacing(); show_panel_iris_fnumber_str();
                                    // show_panel_iris_value();

                                    ImGui::SetCursorScreenPos(p);
                                    is_press |= ImGui::InvisibleButton("##IRIS", ImVec2(-1, -1), ImGuiButtonFlags_MouseButtonLeft);

                                    if (is_press) {
                                        stat_main = em_State::IRIS;
                                    }
                                }
                                ImGui::EndChild();
                            }

                            ///////////////////////////////////////////////////////////////////////
                            // BOTTOM RIGHT.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(2);
                            {
                                if (ImGui::BeginChild("main_bottom_right", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    auto p = ImGui::GetCursorScreenPos();

                                    ImGui::SetCursorScreenPos(ImVec2(right_x, p.y));
                                    auto is_press = (ImGui::Button("WB", btn_size) || ImGui::IsKeyPressed(ImGuiKey_V, false));

                                    spacing(); show_panel_wb_mode_str();
                                    spacing(); show_panel_wb_value();

                                    ImGui::SetCursorScreenPos(p);
                                    is_press |= ImGui::InvisibleButton("##WB", ImVec2(-1, -1), ImGuiButtonFlags_MouseButtonLeft);

                                    if (is_press) {
                                        stat_main = em_State::WHITE_BALANCE;
                                    }
                                }
                                ImGui::EndChild();
                            }
                        }
                        ImGui::EndTable();
                    }
                    ImGui::SetWindowFontScale(1.0f);
                }
            }
            ImGui::EndTable();
        }

        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            // ret = false; // disable ESC.
        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }

        auto [is_drag_left, mouse_delta] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
        if (is_drag_left) {
            ret = false;
        }

        ImGui::PopID();

        return ret;
    }



    /////////////////////////////////////////////////////////////////
    // focus control.
    /////////////////////////////////////////////////////////////////
    void show_panel_focus_control()
    {
        ImGui::PushID("Focus_Control");

        if (ImGui::Button("Focus")) {
            ;
        }

        ImGui::PopID();
    }



    /////////////////////////////////////////////////////////////////
    // IRIS control.
    /////////////////////////////////////////////////////////////////
    void show_panel_iris_mode_str(bool center = false)
    {
        ImGui::PushID("IRIS_MODE_STR");

        auto idx = cgi->get_iris_mode_state();
        auto &vec = iris_mode_state;
        show_panel_state_mode_str(idx, vec, center);

        ImGui::PopID();
    }

    void show_panel_iris_fnumber_str(bool center = false)
    {
        ImGui::PushID("IRIS_FNUMBER_STR");

        auto &imaging = cgi->inquiry_imaging();
        auto fnumber = imaging.ExposureFNumber;
        auto fn_i = fnumber / 100;
        auto fn_f = fnumber % 100;
        auto fn_f_10 = fnumber % 10;
        std::string str = "F" + std::to_string(fn_i);
        if (fn_f != 0) {
            str += ".";
            if (fn_f_10 != 0) {
                str += std::to_string(fn_f);
            } else {
                str += std::to_string(fn_f / 10);
            }
        }

        if (center) {
            centering_text_pos(str.c_str());
        }

        ImGui::Text("%s", str.c_str());

        ImGui::PopID();
    }

    void show_panel_iris_value(bool center = false)
    {
        ImGui::PushID("IRIS_VALUE_STR");

        auto &imaging = cgi->inquiry_imaging();
        auto iris = imaging.ExposureIris;

        auto f = CGICmd::calc_fnum(iris);
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << f;
        std::string str = "F" + ss.str();

        if (center) {
            centering_text_pos(str.c_str());
        }

        ImGui::Text("%s", str.c_str());

        ImGui::PopID();
    }

    void show_panel_iris_mode_auto_manual()
    {
        auto &imaging = cgi->inquiry_imaging();
        auto &on_off = imaging.ExposureAutoIris;

        auto f = [&](CGICmd::COMMON::em_OnOff val) -> void { cgi->set_imaging_ExposureAutoIris(val); };

        show_panel_select_value_listbox(
            "##EXPOSURE_AUTO_IRIS",
            on_off,
            CGICmd::COMMON::OFF,
            CGICmd::COMMON::ON,
            iris_auto_on_off, f,
            0.5f, 0.1f
        );
    }

    void show_panel_iris_mode()
    {
        ImGui::PushID("IRIS_MODE");

        auto &imaging = cgi->inquiry_imaging();

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("iris_mode", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);

                ImGui::TableSetColumnIndex(1);
                show_panel_iris_mode_auto_manual();

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        auto [is_drag_left, mouse_delta] = is_mouse_drag_to_left(ImGuiMouseButton_Left);

        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || is_drag_left) {
            stat_iris = em_StateIris::CONTROL;

        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }

        ImGui::PopID();
    }

    void show_panel_iris_control_auto()
    {
        auto &imaging = cgi->inquiry_imaging();
        auto fnumer = imaging.ExposureFNumber;
        auto iris = imaging.ExposureIris;

        {
            auto p = ImGui::GetCursorScreenPos();
            auto sz = ImGui::GetWindowSize();
            auto text_height = ImGui::GetTextLineHeightWithSpacing();
            auto p_center = ImVec2(p.x, p.y + sz.y / 2 - text_height * 2.25);
            ImGui::SetCursorScreenPos(p_center);
        }
        ImGui::SetWindowFontScale(1.5f);
        show_panel_iris_mode_str(true);
        show_panel_iris_fnumber_str(true);
        show_panel_iris_value(true);
        ImGui::SetWindowFontScale(1.0f);
    }

    iris_list_t gen_iris_list(int idx_max, int idx_min)
    {
        iris_list_t lst_remake;
        auto &lst = exposure_iris;
        auto lst_max = lst.front().first;
        auto lst_min = lst.back().first;

        if (idx_max > lst_max) {
            ;
        }

        {
            auto idx_search = [](auto &lst, auto idx) -> auto {
                auto itr = std::find_if(lst.begin(), lst.end(), [&idx](auto &e) {
                    auto &[k, v] = e;
                    float idx_f = idx;
                    auto step = std::abs(CGICmd::iris_k_add) / 2.0f;
                    bool ret = idx_f <= (k + step) && idx_f > (k - step);
                    return ret;
                });
                return itr;
            };
            auto itr_s = idx_search(lst, idx_max);
            auto itr_e = idx_search(lst, idx_min);
            itr_e++;

            lst_remake.assign(itr_s, itr_e);
        }

        if (idx_min < lst_min) {
            ;
        }

        return lst_remake;
    }

    void show_panel_iris_control_manual()
    {
        auto &imaging = cgi->inquiry_imaging();
        auto &iris = imaging.ExposureIris;
        auto iris_max = imaging.ExposureIrisRange.min;  // step is minus.
        auto iris_min = imaging.ExposureIrisRange.max;  // step is minus.

        {
            float iris_f = iris / std::abs(CGICmd::iris_k_add);
            iris_f = std::round(iris_f);
            iris = std::round(iris_f * std::abs(CGICmd::iris_k_add));
        }

        auto lst = gen_iris_list(iris_max, iris_min);

        auto f = [&](int val) -> void { cgi->set_imaging_ExposureIris(val); };

        const auto [ exposure_iris_max, min_str ] = lst.front();
        const auto [ exposure_iris_min, max_str ] = lst.back();

        show_panel_select_value_listbox(
            "##EXPOSURE_IRIS",
            iris,
            exposure_iris_min,
            exposure_iris_max,
            lst, f,
            0.5f, 0.1f
        );
    }

    void show_panel_iris_control_sub(CGICmd::em_IrisModeState state)
    {
        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("iris_control_sub", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                show_panel_text_select_mode();

                ImGui::TableSetColumnIndex(1);
                switch (state) {
                case CGICmd::em_IrisModeState::AUTO:
                    show_panel_iris_control_auto();
                    break;

                case CGICmd::em_IrisModeState::MANUAL:
                    show_panel_iris_control_manual();
                    break;

                case CGICmd::em_IrisModeState::INVALID:
                default:
                    ;
                }

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        auto [is_drag_left, mouse_delta_left] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
        auto [is_drag_right, mouse_delta_right] = is_mouse_drag_to_right(ImGuiMouseButton_Left);

        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || is_drag_left) {
            stat_main = em_State::MAIN;
        } else if (ImGui::IsKeyPressed(ImGuiKey_X, false) || is_drag_right) {
            stat_iris = em_StateIris::MODE;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }
    }

    void show_panel_iris_control()
    {
        ImGui::PushID("IRIS_Control");

        auto &imaging = cgi->inquiry_imaging();
        auto state = cgi->get_iris_mode_state();

        iris_mode_state_bkup = state;

        switch (stat_iris) {
        case em_StateIris::MODE:
            show_panel_iris_mode();
            break;

        case em_StateIris::CONTROL:
            show_panel_iris_control_sub(state);
            break;

        default:
            ;
        }

        ImGui::PopID();
    }



    /////////////////////////////////////////////////////////////////
    // ISO control.
    /////////////////////////////////////////////////////////////////
    void show_panel_iso_mode_str(bool center = false)
    {
        ImGui::PushID("ISO_MODE_STR");

        auto idx = cgi->get_iso_mode_state();
        auto &vec = iso_mode_state;
        show_panel_state_mode_str(idx, vec, center);

        ImGui::PopID();
    }

    void show_panel_iso_agc_str(bool center = false)
    {
        ImGui::PushID("ISO_AGC_STR");

        auto &imaging = cgi->inquiry_imaging();
        auto idx = imaging.ExposureAGCEnable;
        auto &vec = iso_agc_on_off;
        show_panel_state_mode_str(idx, vec, center);

        ImGui::PopID();
    }

    void show_panel_iso_base_sensitivity_str(bool center = false)
    {
        ImGui::PushID("ISO_BASE_SENSITIVITY_STR");

        auto &imaging = cgi->inquiry_imaging();
        auto idx = imaging.ExposureBaseSensitivity;
        auto &vec = iso_base_sensitivity;
        show_panel_state_mode_str(idx, vec, center);

        ImGui::PopID();
    }

    void show_panel_iso_base_iso_str(bool center = false)
    {
        ImGui::PushID("ISO_BASE_ISO_STR");

        auto &imaging = cgi->inquiry_imaging();
        auto idx = imaging.ExposureBaseISO;
        auto &vec = iso_base_iso;
        show_panel_state_mode_str(idx, vec, center);

        ImGui::PopID();
    }

    void show_panel_iso_value(bool center = false)
    {
        ImGui::PushID("ISO_VALUE");

        auto &imaging = cgi->inquiry_imaging();
        auto state = cgi->get_iso_mode_state();
        auto is_agc = imaging.ExposureAGCEnable == CGICmd::COMMON::ON;

        switch (state) {
        case CGICmd::em_ISOModeState::GAIN:
            {
                auto idx = is_agc ? imaging.ExposureGainTemporary : imaging.ExposureGain;
                auto &lst = CGICmd::exposure_gain;

                std::string str = get_string_from_pair_list(lst, idx);
                if (center) centering_text_pos(str.c_str());
                ImGui::Text("%s", str.c_str());
            }
            break;

        case CGICmd::em_ISOModeState::ISO:
            {
                auto idx = is_agc ? imaging.ExposureISOTemporary : imaging.ExposureISO;
                auto &lst = CGICmd::exposure_iso;

                std::string str = get_string_from_pair_list(lst, idx);
                if (center) centering_text_pos(str.c_str());
                ImGui::Text("%s", str.c_str());
            }
            break;

        case CGICmd::em_ISOModeState::CINE_EI_QUITCK:
        case CGICmd::em_ISOModeState::CINE_EI:
            {
                auto &imaging = cgi->inquiry_imaging();
                auto &base_iso = imaging.ExposureBaseISO;
                auto base_iso_str = json_conv_enum2str(base_iso);
                auto idx = imaging.ExposureExposureIndex;

                std::string str = "---";
#if __cplusplus == 202002L  // C++20.
                if (CGICmd::exposure_exposure_index.contains(base_iso_str)) {
#else
                auto &vec = CGICmd::exposure_exposure_index;
                if (std::find_if(vec.begin(), vec.end(), [&base_iso_str](auto &e){ return e.first == base_iso_str; }) != vec.end()) {
#endif
                    auto &lst = CGICmd::exposure_exposure_index[base_iso_str];
                    str = get_string_from_pair_list(lst, idx);
                }
                if (center) centering_text_pos(str.c_str());
                ImGui::Text("%s", str.c_str());
            }
            break;

        case CGICmd::em_ISOModeState::INVALID:
        default:
            ;

        }

        ImGui::PopID();
    }

    void show_panel_iso_mode_auto_manual()
    {
        auto &imaging = cgi->inquiry_imaging();
        auto &on_off = imaging.ExposureAGCEnable;

        auto f = [&](CGICmd::COMMON::em_OnOff val) -> void { cgi->set_imaging_ExposureAGCEnable(val); };

        show_panel_select_value_listbox(
            "##EXPOSURE_AGC_ENABLE",
            on_off,
            CGICmd::COMMON::OFF,
            CGICmd::COMMON::ON,
            iso_agc_on_off, f,
            0.5f, 0.1f
        );
    }

    void show_panel_iso_mode_base_sensitivity()
    {
        auto &imaging = cgi->inquiry_imaging();
        auto &low_high = imaging.ExposureBaseSensitivity;

        auto f = [&](CGICmd::COMMON::em_LowHigh val) -> void { cgi->set_imaging_ExposureBaseSensitivity(val); };

        show_panel_select_value_listbox(
            "##EXPOSURE_BASE_SENSITIVITY",
            low_high,
            CGICmd::COMMON::LOW,
            CGICmd::COMMON::HIGH,
            iso_base_sensitivity, f,
            0.5f, 0.1f
        );
    }

    void show_panel_iso_mode_base_iso()
    {
        auto &imaging = cgi->inquiry_imaging();
        auto &base_iso = imaging.ExposureBaseISO;

        auto f = [&](CGICmd::em_ExposureBaseISO val) -> void { cgi->set_imaging_ExposureBaseISO(val); };

        show_panel_select_value_listbox(
            "##EXPOSURE_BASE_ISO",
            base_iso,
            CGICmd::ExposureBaseISO_ISO800,
            CGICmd::ExposureBaseISO_ISO12800,
            iso_base_iso, f,
            0.5f, 0.1f
        );
    }

    void show_panel_iso_mode()
    {
        ImGui::PushID("ISO_MODE");

        auto &imaging = cgi->inquiry_imaging();

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("iso_mode", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);

                ImGui::TableSetColumnIndex(1);
                if (stat_iso == em_StateISO::MODE_AUTO_MANUAL) {
                    show_panel_iso_mode_auto_manual();
                } else if (stat_iso == em_StateISO::MODE_BASE_SENSITIVITY || stat_iso == em_StateISO::MODE_BASE_ISO) {
                    auto iso_gain = imaging.ExposureISOGainMode;
                    if (iso_gain == CGICmd::ExposureISOGainMode_GAIN) {
                        show_panel_iso_mode_base_sensitivity();
                    } else if (iso_gain == CGICmd::ExposureISOGainMode_ISO) {
                        show_panel_iso_mode_base_iso();
                    }
                }

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        auto [is_drag_left, mouse_delta] = is_mouse_drag_to_left(ImGuiMouseButton_Left);

        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || is_drag_left) {
            stat_iso = em_StateISO::CONTROL;

        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }

        ImGui::PopID();
    }

    void show_panel_iso_control_gain()
    {
        auto &imaging = cgi->inquiry_imaging();
        auto &gain = imaging.ExposureGain;
        auto is_agc = imaging.ExposureAGCEnable == CGICmd::COMMON::ON;

        if (is_agc) {
            {
                auto p = ImGui::GetCursorScreenPos();
                auto sz = ImGui::GetWindowSize();
                auto text_height = ImGui::GetTextLineHeightWithSpacing();
                auto p_center = ImVec2(p.x, p.y + sz.y / 2 - text_height * 3);
                ImGui::SetCursorScreenPos(p_center);
            }
            ImGui::SetWindowFontScale(1.5f);
            show_panel_iso_mode_str(true);
            show_panel_iso_agc_str(true);
            show_panel_iso_base_sensitivity_str(true);
            show_panel_iso_value(true);
            ImGui::SetWindowFontScale(1.0f);

        } else {
            auto f = [&](int val) -> void { cgi->set_imaging_ExposureGain(val); };

            const auto [ exposure_gain_min, min_str ] = CGICmd::exposure_gain.front();
            const auto [ exposure_gain_max, max_str ] = CGICmd::exposure_gain.back();

            show_panel_select_value_listbox(
                "##EXPOSURE_GAIN",
                gain,
                exposure_gain_min,
                exposure_gain_max,
                CGICmd::exposure_gain, f,
                0.5f, 0.1f
            );
        }
    }

    void show_panel_iso_control_iso()
    {
        auto &imaging = cgi->inquiry_imaging();
        auto &iso = imaging.ExposureISO;
        auto is_agc = imaging.ExposureAGCEnable == CGICmd::COMMON::ON;

        if (is_agc) {
            {
                auto p = ImGui::GetCursorScreenPos();
                auto sz = ImGui::GetWindowSize();
                auto text_height = ImGui::GetTextLineHeightWithSpacing();
                auto p_center = ImVec2(p.x, p.y + sz.y / 2 - text_height * 3);
                ImGui::SetCursorScreenPos(p_center);
            }
            ImGui::SetWindowFontScale(1.5f);
            show_panel_iso_mode_str(true);
            show_panel_iso_agc_str(true);
            show_panel_iso_base_iso_str(true);
            show_panel_iso_value(true);
            ImGui::SetWindowFontScale(1.0f);

        } else {
            auto f = [&](int val) -> void { cgi->set_imaging_ExposureISO(val); };

            const auto [ exposure_iso_min, min_str ] = CGICmd::exposure_iso.front();
            const auto [ exposure_iso_max, max_str ] = CGICmd::exposure_iso.back();

            show_panel_select_value_listbox(
                "##EXPOSURE_ISO",
                iso,
                exposure_iso_min,
                exposure_iso_max,
                CGICmd::exposure_iso, f,
                0.5f, 0.1f
            );
        }
    }

    void show_panel_iso_control_EI()
    {
        auto &imaging = cgi->inquiry_imaging();
        auto &base_iso = imaging.ExposureBaseISO;
        auto base_iso_str = json_conv_enum2str(base_iso);

#if __cplusplus == 202002L  // C++20.
        auto &lst = (CGICmd::exposure_exposure_index.contains(base_iso_str))
#else
        auto &vec = CGICmd::exposure_exposure_index;
        auto &lst = (std::find_if(vec.begin(), vec.end(), [&base_iso_str](auto &e){ return e.first == base_iso_str; }) != vec.end())
#endif
            ? CGICmd::exposure_exposure_index[base_iso_str]
            : CGICmd::exposure_exposure_index_iso800
            ;

        auto &EI = imaging.ExposureExposureIndex;

        auto f = [&](int val) -> void { cgi->set_imaging_ExposureExposureIndex(val); };

        const auto [ EI_min, min_str ] = lst.front();
        const auto [ EI_max, max_str ] = lst.back();

        show_panel_select_value_listbox(
            "##EXPOSURE_EXPOSSSURE_INDEX",
            EI,
            EI_min,
            EI_max,
            lst, f,
            0.5f, 0.1f
        );
    }

    void show_panel_iso_control_sub(CGICmd::em_ISOModeState state)
    {
        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("iso_control_sub", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                switch (state) {
                case CGICmd::em_ISOModeState::GAIN:
                case CGICmd::em_ISOModeState::ISO:
                    show_panel_text_select_mode_right_up_down();
                    break;

                case CGICmd::em_ISOModeState::CINE_EI:
                    show_panel_text_select_mode_right_down();
                    break;

                case CGICmd::em_ISOModeState::CINE_EI_QUITCK:
                case CGICmd::em_ISOModeState::INVALID:
                default:
                    ;
                }

                ImGui::TableSetColumnIndex(1);
                switch (state) {
                case CGICmd::em_ISOModeState::GAIN:
                    show_panel_iso_control_gain();
                    break;

                case CGICmd::em_ISOModeState::ISO:
                    show_panel_iso_control_iso();
                    break;

                case CGICmd::em_ISOModeState::CINE_EI_QUITCK:
                case CGICmd::em_ISOModeState::CINE_EI:
                    show_panel_iso_control_EI();
                    break;

                case CGICmd::em_ISOModeState::INVALID:
                default:
                    ;
                }

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        auto [is_drag_left, mouse_delta_left] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
        auto [is_drag_right, mouse_delta_right] = is_mouse_drag_to_right(ImGuiMouseButton_Left);

        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || is_drag_left) {
            stat_main = em_State::MAIN;
        } else if (ImGui::IsKeyPressed(ImGuiKey_X, false) || (is_drag_right && mouse_delta_right.y < 0.0f)) {
            if (state == CGICmd::em_ISOModeState::GAIN
                || state == CGICmd::em_ISOModeState::ISO
                ) {
                stat_iso = em_StateISO::MODE_AUTO_MANUAL;
            }
        } else if (ImGui::IsKeyPressed(ImGuiKey_W, false) || (is_drag_right)) {
            if (state == CGICmd::em_ISOModeState::GAIN
                || state == CGICmd::em_ISOModeState::ISO
                || state == CGICmd::em_ISOModeState::CINE_EI
                ) {
                stat_iso = em_StateISO::MODE_BASE_SENSITIVITY;  // = MODE_BASE_ISO.
            }
        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }
    }

    void show_panel_iso_control()
    {
        ImGui::PushID("ISO_Control");

        auto &imaging = cgi->inquiry_imaging();
        auto state = cgi->get_iso_mode_state();

        iso_mode_state_bkup = state;

        switch (stat_iso) {
        case em_StateISO::MODE_AUTO_MANUAL:
        case em_StateISO::MODE_BASE_SENSITIVITY:    // = MODE_BASE_ISO.
            show_panel_iso_mode();
            break;

        case em_StateISO::CONTROL:
            show_panel_iso_control_sub(state);
            break;

        default:
            ;
        }

        ImGui::PopID();
    }



    /////////////////////////////////////////////////////////////////
    // shutter control.
    /////////////////////////////////////////////////////////////////
    void show_panel_shutter_mode_str(bool center = false)
    {
        ImGui::PushID("SHUTTER_MODE_STR");

        auto &imaging = cgi->inquiry_imaging();
        auto idx = imaging.ExposureShutterModeState;
        auto &vec = exposure_shutter_mode_state;
        show_panel_state_mode_str(idx, vec, center);

        ImGui::PopID();
    }

    void show_panel_shutter_value(bool center = false)
    {
        ImGui::PushID("SHUTTER_VALUE");

        auto &imaging = cgi->inquiry_imaging();
        auto &state = imaging.ExposureShutterModeState;

        switch (state) {
        case CGICmd::ExposureShutterModeState_SPEED:
        case CGICmd::ExposureShutterModeState_AUTO:
            {
                auto &project = cgi->inquiry_project();
                auto frame_rate = json_conv_enum2str(project.RecFormatFrequency);
                auto idx = imaging.ExposureExposureTime;

                std::string str = "---";
#if __cplusplus == 202002L  // C++20.
                if (CGICmd::exposure_exposure_time.contains(frame_rate)) {
#else
                auto &vec = CGICmd::exposure_exposure_time;
                if (std::find_if(vec.begin(), vec.end(), [&frame_rate](auto &e){ return e.first == frame_rate; }) != vec.end()) {
#endif
                    auto &lst = CGICmd::exposure_exposure_time[frame_rate];
                    str = get_string_from_pair_list(lst, idx);
                }
                if (center) centering_text_pos(str.c_str());
                ImGui::Text("%s", str.c_str());
            }
            break;

        case CGICmd::ExposureShutterModeState_ANGLE:
            {
                auto idx = imaging.ExposureAngle;
                auto &lst = CGICmd::exposure_angle;

                std::string str = get_string_from_pair_list(lst, idx);
                if (center) centering_text_pos(str.c_str());
                ImGui::Text("%s", str.c_str());
            }
            break;

        case CGICmd::ExposureShutterModeState_ECS:
            {
                auto idx = imaging.ExposureECS;
                auto val = imaging.ExposureECSValue;
                auto val_f = val / 1000.0f;
                if (center) centering_text_pos("123456789012");
                ImGui::Text("[%d] %.2f", idx, val_f);
            }
            break;

        case CGICmd::ExposureShutterModeState_OFF:
        default:
            ;
        }

        ImGui::PopID();
    }

    void show_panel_shutter_mode()
    {
        ImGui::PushID("SHUTTER_MODE");

        auto &imaging = cgi->inquiry_imaging();
        auto &state = imaging.ExposureShutterModeState;

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("shutter_mode", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);

                ImGui::TableSetColumnIndex(1);
                {
                    auto f = [&](CGICmd::em_ExposureShutterModeState val) -> void { cgi->set_imaging_ExposureShutterModeState(val); };

                    show_panel_select_value_listbox(
                        "##EXPOSURE_SHUTTER_MODE_STATE",
                        state,
                        CGICmd::ExposureShutterModeState_OFF,
                        CGICmd::ExposureShutterModeState_AUTO,
                        exposure_shutter_mode_state, f,
                        0.5f, 0.1f
                    );
                }

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            stat_shutter = em_StateShutter::CONTROL;

        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }

        auto [is_drag_left, mouse_delta] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
        if (is_drag_left) {
            stat_shutter = em_StateShutter::CONTROL;
        }

        ImGui::PopID();
    }

    void show_panel_shutter_auto()
    {
        ImGui::PushID("SHUTTER_AUTO");

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("shutter_auto", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                show_panel_text_select_mode();

                ImGui::TableSetColumnIndex(1);
                {
                    auto p = ImGui::GetCursorScreenPos();
                    auto sz = ImGui::GetWindowSize();
                    auto text_height = ImGui::GetTextLineHeightWithSpacing();
                    auto p_center = ImVec2(p.x, p.y + sz.y / 2 - text_height * 1.5f);
                    ImGui::SetCursorScreenPos(p_center);
                }
                ImGui::SetWindowFontScale(1.5f);
                show_panel_shutter_mode_str(true);
                show_panel_shutter_value(true);
                ImGui::SetWindowFontScale(1.0f);

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        ImGui::PopID();
    }

    void show_panel_shutter_speed()
    {
        ImGui::PushID("SHUTTER_SPEED");

        auto &imaging = cgi->inquiry_imaging();

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("shutter_speed", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                show_panel_text_select_mode();

                ImGui::TableSetColumnIndex(1);
                {
                    auto &project = cgi->inquiry_project();
                    auto frame_rate = json_conv_enum2str(project.RecFormatFrequency);
#if __cplusplus == 202002L  // C++20.
                    auto &lst = (CGICmd::exposure_exposure_time.contains(frame_rate))
#else
                    auto &vec = CGICmd::exposure_exposure_time;
                    auto &lst = (std::find_if(vec.begin(), vec.end(), [&frame_rate](auto &e){ return e.first == frame_rate; }) != vec.end())
#endif
                        ? CGICmd::exposure_exposure_time[frame_rate]
                        : CGICmd::exposure_exposure_time_5994p
                        ;
                    auto f = [&](int val) -> void { cgi->set_imaging_ExposureExposureTime(val); };

                    show_panel_select_value_listbox(
                        "##EXPOSURE_EXPOSURE_TIME",
                        imaging.ExposureExposureTime,
                        imaging.ExposureExposureTimeRange.min,
                        imaging.ExposureExposureTimeRange.max,
                        lst, f,
                        0.5f, 0.1f
                    );
                }

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        ImGui::PopID();
    }

    void show_panel_shutter_angle()
    {
        ImGui::PushID("SHUTTER_ANGLE");

        auto &imaging = cgi->inquiry_imaging();

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("shutter_angle", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                show_panel_text_select_mode();

                ImGui::TableSetColumnIndex(1);
                {
                    auto &lst = CGICmd::exposure_angle;
                    auto f = [&](int val) -> void { cgi->set_imaging_ExposureAngle(val); };

                    show_panel_select_value_listbox(
                        "##EXPOSURE_ANGLE",
                        imaging.ExposureAngle,
                        imaging.ExposureAngleRange.min,
                        imaging.ExposureAngleRange.max,
                        lst, f,
                        0.5f, 0.1f
                    );
                }

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        ImGui::PopID();
    }

    void show_panel_shutter_ecs()
    {
        ImGui::PushID("SHUTTER_ECS");

        auto &imaging = cgi->inquiry_imaging();

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("shutter_ecs", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                show_panel_text_select_mode();

                ImGui::TableSetColumnIndex(1);
                {
                    std::list<std::pair<int, std::string>> lst;
                    auto idx_min = imaging.ExposureECSRange.min;
                    auto idx_max = imaging.ExposureECSRange.max;
                    for (auto i = idx_min; i <= idx_max; i++) {
                        lst.emplace_back(std::pair<int, std::string>{ i, std::to_string(i) });
                    }

                    auto f = [&](int val) -> void { cgi->set_imaging_ExposureECS(val); };

                    show_panel_select_value_listbox(
                        "##EXPOSURE_ECS",
                        imaging.ExposureECS,
                        imaging.ExposureECSRange.min,
                        imaging.ExposureECSRange.max,
                        lst, f,
                        0.275f, 0.050f
                    );
                }

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        ImGui::PopID();
    }

    void show_panel_shutter_off()
    {
        ImGui::PushID("SHUTTER_OFF");

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("shutter_off", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                show_panel_text_select_mode();

                ImGui::TableSetColumnIndex(1);
                {
                    auto p = ImGui::GetCursorScreenPos();
                    auto sz = ImGui::GetWindowSize();
                    auto text_height = ImGui::GetTextLineHeightWithSpacing();
                    auto p_center = ImVec2(p.x, p.y + sz.y / 2 - text_height * 1.5f);
                    ImGui::SetCursorScreenPos(p_center);
                }
                ImGui::SetWindowFontScale(1.5f);
                show_panel_shutter_mode_str(true);
                show_panel_shutter_value(true);
                ImGui::SetWindowFontScale(1.0f);

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        ImGui::PopID();
    }

    void show_panel_shutter_control()
    {
        ImGui::PushID("Shutter_Control");

        auto &imaging = cgi->inquiry_imaging();
        auto &state = imaging.ExposureShutterModeState;

        shutter_mode_state_bkup = state;

        if (stat_shutter == em_StateShutter::MODE) {
            show_panel_shutter_mode();

        } else if (stat_shutter == em_StateShutter::CONTROL) {
            switch (state) {
            case CGICmd::ExposureShutterModeState_SPEED:
                show_panel_shutter_speed();
                break;

            case CGICmd::ExposureShutterModeState_ANGLE:
                show_panel_shutter_angle();
                break;

            case CGICmd::ExposureShutterModeState_ECS:
                show_panel_shutter_ecs();
                break;

            case CGICmd::ExposureShutterModeState_AUTO:
                show_panel_shutter_auto();
                break;

            case CGICmd::ExposureShutterModeState_OFF:
                show_panel_shutter_off();
                break;

            default:
                stat_main = em_State::MAIN;
            }

            if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                stat_main = em_State::MAIN;
            } else if (ImGui::IsKeyPressed(ImGuiKey_X, false)) {
                stat_shutter = em_StateShutter::MODE;
            } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
                stat_main_bkup = stat_main;
                stat_main = em_State::LIVE_VIEW;
            }

            auto [is_drag_left, mouse_delta_left] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
            auto [is_drag_right, mouse_delta_right] = is_mouse_drag_to_right(ImGuiMouseButton_Left);
            if (is_drag_left) {
                stat_main = em_State::MAIN;
            } else if (is_drag_right) {
                stat_shutter = em_StateShutter::MODE;
            }
        }

        ImGui::PopID();
    }



    /////////////////////////////////////////////////////////////////
    // white_balance control.
    /////////////////////////////////////////////////////////////////
    void show_panel_wb_mode_str(bool center = false)
    {
        ImGui::PushID("WB_MODE_STR");

        auto &imaging = cgi->inquiry_imaging();
        auto idx = cgi->get_wb_mode_state();
        auto &vec = white_balance_mode_state;
        show_panel_state_mode_str(idx, vec, center);

        ImGui::PopID();
    }

    void show_panel_wb_value(bool center = false)
    {
        ImGui::PushID("WB_VALUE");

        auto &imaging = cgi->inquiry_imaging();
        auto state = cgi->get_wb_mode_state();

        switch (state) {
        case CGICmd::em_WhiteBalanceModeState::AUTO:
            {
                auto val = imaging.WhiteBalanceColorTempCurrent;
                std::string str = std::to_string(val) + "(K)";

                if (center) centering_text_pos(str.c_str());
                ImGui::Text("%s", str.c_str());
            }
            break;

        case CGICmd::em_WhiteBalanceModeState::MANUAL:
            {
                auto idx = imaging.WhiteBalanceColorTemp;
                auto &lst = CGICmd::white_balance_color_temp;

                std::string str = get_string_from_pair_list(lst, idx);
                if (center) centering_text_pos(str.c_str());
                ImGui::Text("%s", str.c_str());
            }
            break;

        default:
            ;
        }

        ImGui::PopID();
    }

    void show_panel_wb_mode()
    {
        ImGui::PushID("WB_MODE");

        auto &imaging = cgi->inquiry_imaging();
        auto state = cgi->get_wb_mode_state();

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("wb_mode", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);

                ImGui::TableSetColumnIndex(1);
                {
                    auto f = [&](CGICmd::em_WhiteBalanceModeState val) -> void { cgi->set_imaging_WhiteBalanceModeState(val); };

                    show_panel_select_value_listbox(
                        "##WHITE_BALANCE_MODE_STATE",
                        state,
                        CGICmd::em_WhiteBalanceModeState::AUTO,
                        CGICmd::em_WhiteBalanceModeState::MANUAL,
                        white_balance_mode_state, f,
                        0.5f, 0.1f
                    );
                }

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            stat_wb = em_StateWhiteBalance::CONTROL;

        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }

        auto [is_drag_left, mouse_delta] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
        if (is_drag_left) {
            stat_wb = em_StateWhiteBalance::CONTROL;
        }

        ImGui::PopID();
    }

    void show_panel_wb_auto()
    {
        ImGui::PushID("WB_AUTO");

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("wb_auto", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                show_panel_text_select_mode();

                ImGui::TableSetColumnIndex(1);
                {
                    auto p = ImGui::GetCursorScreenPos();
                    auto sz = ImGui::GetWindowSize();
                    auto text_height = ImGui::GetTextLineHeightWithSpacing();
                    auto p_center = ImVec2(p.x, p.y + sz.y / 2 - text_height * 1.5f);
                    ImGui::SetCursorScreenPos(p_center);
                }
                ImGui::SetWindowFontScale(1.5f);
                show_panel_wb_mode_str(true);
                show_panel_wb_value(true);
                ImGui::SetWindowFontScale(1.0f);

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        ImGui::PopID();
    }

    void show_panel_wb_color_temp()
    {
        ImGui::PushID("WB_COLOR_TEMP");

        auto &imaging = cgi->inquiry_imaging();

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("wb_color_temp", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                show_panel_text_select_mode();

                ImGui::TableSetColumnIndex(1);
                {
                    auto &lst = CGICmd::white_balance_color_temp;
                    auto f = [&](int val) -> void { cgi->set_imaging_WhiteBalanceColorTemp(val); };

                    auto &[k_min, v_min] = lst.front();
                    auto &[k_max, v_max] = lst.back();

                    show_panel_select_value_listbox(
                        "##WB_COLOR_TEMP",
                        imaging.WhiteBalanceColorTemp,
                        k_min,
                        k_max,
                        lst, f,
                        0.5f, 0.1f
                    );
                }

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        ImGui::PopID();
    }

    void show_panel_white_balance_control()
    {
        ImGui::PushID("White_Balance_Control");

        auto &imaging = cgi->inquiry_imaging();
        auto state = cgi->get_wb_mode_state();

        wb_mode_state_bkup = state;

        if (stat_wb == em_StateWhiteBalance::MODE) {
            show_panel_wb_mode();

        } else if (stat_wb == em_StateWhiteBalance::CONTROL) {
            switch (state) {
            case CGICmd::em_WhiteBalanceModeState::AUTO:
                show_panel_wb_auto();
                break;

            case CGICmd::em_WhiteBalanceModeState::MANUAL:
                show_panel_wb_color_temp();
                break;

            case CGICmd::em_WhiteBalanceModeState::INVALID:
                // set to AUTO mode.
                state = CGICmd::em_WhiteBalanceModeState::AUTO;
                cgi->set_imaging_WhiteBalanceModeState(state);
                cgi->set_wb_mode_state(state);
                break;

            default:
                stat_main = em_State::MAIN;
            }

            if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                stat_main = em_State::MAIN;
            } else if (ImGui::IsKeyPressed(ImGuiKey_X, false)) {
                stat_wb = em_StateWhiteBalance::MODE;
            } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
                stat_main_bkup = stat_main;
                stat_main = em_State::LIVE_VIEW;
            }

            auto [is_drag_left, mouse_delta_left] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
            auto [is_drag_right, mouse_delta_right] = is_mouse_drag_to_right(ImGuiMouseButton_Left);
            if (is_drag_left) {
                stat_main = em_State::MAIN;
            } else if (is_drag_right) {
                stat_wb = em_StateWhiteBalance::MODE;
            }
        }

        ImGui::PopID();
    }



    /////////////////////////////////////////////////////////////////
    // ND control.
    /////////////////////////////////////////////////////////////////
    void show_panel_nd_mode_str(bool center = false)
    {
        ImGui::PushID("ND_MODE_STR");

        auto idx = cgi->get_nd_mode_state();
        auto &vec = nd_mode_state;
        show_panel_state_mode_str(idx, vec, center);

        ImGui::PopID();
    }

    void show_panel_nd_value(bool center = false)
    {
        ImGui::PushID("ND_VALUE_STR");

        auto &imaging = cgi->inquiry_imaging();
        auto idx = imaging.ExposureNDVariable;
        auto &vec = CGICmd::exposure_nd;
        show_panel_state_mode_str(idx, vec, center);

        ImGui::PopID();
    }

    void show_panel_nd_mode_select()
    {
        auto &imaging = cgi->inquiry_imaging();
        auto mode = cgi->get_nd_mode_state();

        auto f = [&](CGICmd::em_NDModeState val) -> void { cgi->set_nd_mode_state(val); };

        show_panel_select_value_listbox(
            "##ND_MODE_SELECT",
            mode,
            CGICmd::em_NDModeState::AUTO,
            CGICmd::em_NDModeState::CLEAR,
            nd_mode_state, f,
            0.5f, 0.1f
        );
    }

    void show_panel_nd_mode()
    {
        ImGui::PushID("ND_MODE");

        auto &imaging = cgi->inquiry_imaging();

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("nd_mode", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);

                ImGui::TableSetColumnIndex(1);
                show_panel_nd_mode_select();

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        auto [is_drag_left, mouse_delta] = is_mouse_drag_to_left(ImGuiMouseButton_Left);

        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || is_drag_left) {
            stat_nd = em_StateND::CONTROL;

        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }

        ImGui::PopID();
    }

    void show_panel_nd_control_auto()
    {
        {
            auto p = ImGui::GetCursorScreenPos();
            auto sz = ImGui::GetWindowSize();
            auto text_height = ImGui::GetTextLineHeightWithSpacing();
            auto p_center = ImVec2(p.x, p.y + sz.y / 2 - text_height * 1.5f);
            ImGui::SetCursorScreenPos(p_center);
        }
        ImGui::SetWindowFontScale(1.5f);
        show_panel_nd_mode_str(true);
        show_panel_nd_value(true);
        ImGui::SetWindowFontScale(1.0f);
    }

    void show_panel_nd_control_manual()
    {
        auto &imaging = cgi->inquiry_imaging();
        auto &nd = imaging.ExposureNDVariable;
        const auto &lst = CGICmd::exposure_nd;
        auto nd_min = lst.front().first;
        auto nd_max = lst.back().first;

        auto f = [&](int val) -> void { cgi->set_imaging_ExposureNDVariable(val); };

        show_panel_select_value_listbox(
            "##EXPOSURE_ND_VARIABLE",
            nd,
            nd_min,
            nd_max,
            lst, f,
            0.5f, 0.1f
        );
    }

    void show_panel_nd_control_clear()
    {
        {
            auto p = ImGui::GetCursorScreenPos();
            auto sz = ImGui::GetWindowSize();
            auto text_height = ImGui::GetTextLineHeightWithSpacing();
            auto p_center = ImVec2(p.x, p.y + sz.y / 2 - text_height * 0.75f);
            ImGui::SetCursorScreenPos(p_center);
        }
        ImGui::SetWindowFontScale(1.5f);
        show_panel_nd_mode_str(true);
        ImGui::SetWindowFontScale(1.0f);
    }

    void show_panel_nd_control_sub(CGICmd::em_NDModeState state)
    {
        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("nd_control_sub", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                show_panel_text_select_mode();

                ImGui::TableSetColumnIndex(1);
                switch (state) {
                case CGICmd::em_NDModeState::AUTO:
                    show_panel_nd_control_auto();
                    break;

                case CGICmd::em_NDModeState::MANUAL:
                    show_panel_nd_control_manual();
                    break;

                case CGICmd::em_NDModeState::CLEAR:
                    show_panel_nd_control_clear();
                    break;

                case CGICmd::em_NDModeState::INVALID:
                default:
                    ;
                }

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        auto [is_drag_left, mouse_delta_left] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
        auto [is_drag_right, mouse_delta_right] = is_mouse_drag_to_right(ImGuiMouseButton_Left);

        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || is_drag_left) {
            stat_main = em_State::MAIN;
        } else if (ImGui::IsKeyPressed(ImGuiKey_X, false) || is_drag_right) {
            stat_nd = em_StateND::MODE;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }
    }

    void show_panel_nd_control()
    {
        ImGui::PushID("ND_Control");

        auto &imaging = cgi->inquiry_imaging();
        auto state = cgi->get_nd_mode_state();

        nd_mode_state_bkup = state;

        switch (stat_nd) {
        case em_StateND::MODE:
            show_panel_nd_mode();
            break;

        case em_StateND::CONTROL:
            show_panel_nd_control_sub(state);
            break;

        default:
            ;
        }

        ImGui::PopID();
    }



    /////////////////////////////////////////////////////////////////
    // SYSTEM control.
    /////////////////////////////////////////////////////////////////
    void show_panel_system_fps_str(bool center = false)
    {
        ImGui::PushID("SYSTEM_FPS");

        auto &project = cgi->inquiry_project();
        auto idx = project.RecFormatFrequency;
        auto &vec = system_fps;
        show_panel_state_mode_str(idx, vec, center);

        ImGui::PopID();
    }

    void show_panel_system_video_format_str(bool center = false)
    {
        ImGui::PushID("SYSTEM_VIDEO_FORMAT");

        auto &project = cgi->inquiry_project();
        auto idx = project.RecFormatVideoFormat;
        auto &vec = system_video_format;
        show_panel_state_mode_str(idx, vec, center);

        ImGui::PopID();
    }

    void show_panel_system_shooting_mode_str(bool center = false)
    {
        ImGui::PushID("SYSTEM_VIDEO_FORMAT");

        auto &project = cgi->inquiry_project();
        auto idx = project.BaseSettingShootingMode;
        auto &vec = system_shooting_mode;
        show_panel_state_mode_str(idx, vec, center);

        ImGui::PopID();
    }

    void show_panel_system_control_fps_select()
    {
        // auto f = [&](CGICmd::em_RecFormatFrequency val) -> void { cgi->set_project_RecFormatFrequency(val); };
        auto f = [&](CGICmd::em_RecFormatFrequency val) -> void {};  // dummy function.

        show_panel_select_value_listbox(
            "##SYSTEM_Control_FPS_SELECT",
            rec_fps_tmp,
            CGICmd::em_RecFormatFrequency::RecFormatFrequency_5994,
            CGICmd::em_RecFormatFrequency::RecFormatFrequency_2398,
            system_fps, f,
            0.5f, 0.1f
        );
    }

    void show_panel_system_control_fps()
    {
        ImGui::PushID("SYTEM_Control_FPS");

        auto &project = cgi->inquiry_project();

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("SYSTEM_Control_FPS", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                {
                    ImGui::SetWindowFontScale(1.5f);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 0, 1));
                    ImGui::TextWrapped("Push the DIAL BUTTON to set FPS,");
                    ImGui::TextWrapped("then wait for the camera to reboot...");
                    ImGui::PopStyleColor();
                    ImGui::SetWindowFontScale(1.0f);
                }

                ImGui::TableSetColumnIndex(1);
                show_panel_system_control_fps_select();

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        auto [is_drag_left, mouse_delta_left] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
        auto [is_drag_right, mouse_delta_right] = is_mouse_drag_to_right(ImGuiMouseButton_Left);

        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || is_drag_left) {
            stat_system = em_System_State::MAIN;

        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || is_drag_right) {
            cgi->set_project_RecFormatFrequency(rec_fps_tmp);
        }

        ImGui::PopID();
    }

    void show_panel_system_control_shooting_mode_select()
    {
        auto &project = cgi->inquiry_project();
        auto &mode = project.BaseSettingShootingMode;

        auto f = [&](CGICmd::em_BaseSettingShootingMode val) -> void { cgi->set_project_BaseSettingShootingMode(val); };

        show_panel_select_value_listbox(
            "##SYSTEM_Control_SHOOTING_MODE_SELECT",
            mode,
            CGICmd::em_BaseSettingShootingMode::BaseSettingShootingMode_CUSTOM,
            CGICmd::em_BaseSettingShootingMode::BaseSettingShootingMode_CINE_EI_QUICK,
            system_shooting_mode, f,
            0.5f, 0.1f
        );
    }

    void show_panel_system_control_shooting_mode()
    {
        ImGui::PushID("SYTEM_Control_SHOOTING_MODE");

        auto &project = cgi->inquiry_project();

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("SYSTEM_Control_SHOOTING_MODE", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);

                ImGui::TableSetColumnIndex(1);
                show_panel_system_control_shooting_mode_select();

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        auto [is_drag_left, mouse_delta] = is_mouse_drag_to_left(ImGuiMouseButton_Left);

        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || is_drag_left) {
            stat_system = em_System_State::MAIN;

        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }

        ImGui::PopID();
    }

    void show_panel_system_control_video_format_select()
    {
        auto &project = cgi->inquiry_project();
        auto &val = project.RecFormatVideoFormat;
        auto lst = project.RecFormatVideoFormatList.buf;
        std::list<std::pair<CGICmd::em_RecFormatVideoFormat, std::string>> system_video_format_list;
        for (auto &vf : lst) {
            auto &vec = system_video_format;
            auto itr = std::find_if(vec.begin(), vec.end(), [&vf](auto &e){ return e.first == vf; });
            if (itr != vec.end()) {
                system_video_format_list.push_back(*itr);
            }
        }

        auto f = [&](CGICmd::em_RecFormatVideoFormat val) -> void { cgi->set_project_RecFormatVideoFormat(val); };

        show_panel_select_value_listbox(
            "##SYSTEM_Control_VIDEO_FORMAT_SELECT",
            val,
            CGICmd::em_RecFormatVideoFormat::RecFormatVideoFormat_4096x2160p,
            CGICmd::em_RecFormatVideoFormat::RecFormatVideoFormat_1920x1080p_35,
            system_video_format_list, f,
            0.5f, 0.1f
        );
    }

    void show_panel_system_control_video_format()
    {
        ImGui::PushID("SYTEM_Control_VIDEO_FORMAT");

        auto &project = cgi->inquiry_project();

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("SYSTEM_Control_VIDEO_FORMAT", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);

                ImGui::TableSetColumnIndex(1);
                show_panel_system_control_video_format_select();

                ImGui::TableSetColumnIndex(2);
                show_panel_live_view_with_info();
            }
            ImGui::EndTable();
        }

        auto [is_drag_left, mouse_delta] = is_mouse_drag_to_left(ImGuiMouseButton_Left);

        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) || is_drag_left) {
            stat_system = em_System_State::MAIN;

        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }

        ImGui::PopID();
    }

    void show_panel_system_control_main()
    {
        ImGui::PushID("SYSTEM_Control_Main");

        auto tbl_flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        auto cld_flags = ImGuiChildFlags_None;
        auto win_flags = ImGuiWindowFlags_None;

        auto &style = ImGui::GetStyle();
        auto frm_padding = style.FramePadding;
        auto win_size = ImGui::GetWindowSize();

        auto btn_width = win_size.x / 3 - frm_padding.x * 4;
        auto btn_size = ImVec2(btn_width, 0);

        auto left_x = win_size.x / 6 - btn_width / 2;
        auto center_x = left_x + win_size.x / 3;
        auto right_x = center_x + win_size.x / 3;

        constexpr auto font_scale = 1.5f;
        auto spacing = []() { ImGui::Text(" "); ImGui::SameLine(); };

        if (ImGui::BeginTable("system_control", 1, tbl_flags))
        {
            auto p = ImGui::GetCursorScreenPos();
            auto sz = ImGui::GetWindowSize();
            float min_row_height = (sz.y - p.y - 24) / 3;

            for (int row = 0; row < 3; row++)
            {
                ImGui::TableNextRow(ImGuiTableRowFlags_None, min_row_height);
                ImGui::TableNextColumn();

                if (row == 0) {
                    ImGui::SetWindowFontScale(font_scale);
                    if (ImGui::BeginTable("system_control_top", 3, ImGuiTableFlags_BordersInnerV))
                    {
                        for (int row = 0; row < 1; row++)
                        {
                            ImGui::TableNextRow();

                            ///////////////////////////////////////////////////////////////////////
                            // TOP LEFT.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(0);
                            {
                                if (ImGui::BeginChild("system_control_top_left", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    auto p = ImGui::GetCursorScreenPos();

                                    ImGui::SetCursorScreenPos(ImVec2(left_x, p.y));
                                    auto is_press = (ImGui::Button("FPS", btn_size) || ImGui::IsKeyPressed(ImGuiKey_W, false));

                                    spacing(); show_panel_system_fps_str();

                                    ImGui::SetCursorScreenPos(p);
                                    is_press |= ImGui::InvisibleButton("##FPS", ImVec2(-1, -1), ImGuiButtonFlags_MouseButtonLeft);

                                    if (is_press) {
                                        stat_system = em_System_State::FPS;
                                        rec_fps_tmp = cgi->inquiry_project().RecFormatFrequency;
                                    }
                                }
                                ImGui::EndChild();
                            }

                            ///////////////////////////////////////////////////////////////////////
                            // TOP CENTER.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(1);
                            {
                                if (ImGui::BeginChild("system_control_top_center", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    auto p = ImGui::GetCursorScreenPos();

                                    ImGui::SetCursorScreenPos(ImVec2(center_x, p.y));
                                    auto is_press = (ImGui::Button("SHOOT MODE", btn_size) || ImGui::IsKeyPressed(ImGuiKey_E, false));

                                    ImGui::SetWindowFontScale(0.85);
                                    spacing(); show_panel_system_shooting_mode_str();

                                    ImGui::SetCursorScreenPos(p);
                                    is_press |= ImGui::InvisibleButton("##SHOOT MODE", ImVec2(-1, -1), ImGuiButtonFlags_MouseButtonLeft);

                                    if (is_press) {
                                        stat_system = em_System_State::SHOOTING_MODE;
                                    }
                                }
                                ImGui::EndChild();
                            }

                            ///////////////////////////////////////////////////////////////////////
                            // TOP RIGHT.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(2);
                            {
                                if (ImGui::BeginChild("system_control_top_right", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    ;
                                }
                                ImGui::EndChild();
                            }
                        }
                        ImGui::EndTable();
                    }
                    ImGui::SetWindowFontScale(1.0f);

                } else if (row == 1) {
                    if (ImGui::BeginTable("system_control_middle", 2, ImGuiTableFlags_BordersInnerV))
                    {
                        for (int row = 0; row < 1; row++)
                        {
                            ImGui::TableNextRow();

                            ///////////////////////////////////////////////////////////////////////
                            // MIDDLE LEFT.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(0);
                            {
                                ImGui::SetWindowFontScale(0.75f);

                                // print selected camera info.
                                show_panel_print_selected_camera_info();

                                ImGui::SetWindowFontScale(1.0f);
                            }

                            ///////////////////////////////////////////////////////////////////////
                            // MIDDLE RIGHT.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(1);
                            {
                                ImVec2 p = ImGui::GetCursorScreenPos();
                                ImVec2 win_size = ImGui::GetWindowSize();

                                show_panel_live_view(win_size.x / 4, false, true);

                                ImGui::SetCursorScreenPos(p);
                                ImGui::InvisibleButton("##INVISIBULE_BUTTON", ImVec2(win_size.x / 4, min_row_height));
                                auto is_hovered = ImGui::IsItemHovered();
                                if (is_hovered && ImGui::IsMouseClicked(0)) {
                                    stat_main_bkup = stat_main;
                                    stat_main = em_State::LIVE_VIEW;
                                }

                                ImGui::SameLine();

                                // show_panel_blink_tally(false, true);
                                show_panel_blink_tally(remote_server.ip_address);
                            }
                        }
                        ImGui::EndTable();
                    }

                } else if (row == 2) {
                    ImGui::SetWindowFontScale(font_scale);
                    if (ImGui::BeginTable("system_control_bottom", 3, ImGuiTableFlags_BordersInnerV))
                    {
                        for (int row = 0; row < 1; row++)
                        {
                            ImGui::TableNextRow();

                            ///////////////////////////////////////////////////////////////////////
                            // BOTTOM LEFT.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(0);
                            {
                                if (ImGui::BeginChild("system_control_bottom_left", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    auto p = ImGui::GetCursorScreenPos();

                                    ImGui::SetCursorScreenPos(ImVec2(left_x, p.y));
                                    auto is_press = (ImGui::Button("VIDEO Fmt.", btn_size) || ImGui::IsKeyPressed(ImGuiKey_X, false));

                                    spacing(); show_panel_system_video_format_str();

                                    ImGui::SetCursorScreenPos(p);
                                    is_press |= ImGui::InvisibleButton("##VIDEO Fmt.", ImVec2(-1, -1), ImGuiButtonFlags_MouseButtonLeft);

                                    if (is_press) {
                                        stat_system = em_System_State::VIDEO_FORMAT;
                                    }
                                }
                                ImGui::EndChild();
                            }

                            ///////////////////////////////////////////////////////////////////////
                            // BOTTOM CENTER.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(1);
                            {
                                if (ImGui::BeginChild("system_control_bottom_center", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    ;
                                }
                                ImGui::EndChild();
                            }

                            ///////////////////////////////////////////////////////////////////////
                            // BOTTOM RIGHT.
                            ///////////////////////////////////////////////////////////////////////
                            ImGui::TableSetColumnIndex(2);
                            {
                                if (ImGui::BeginChild("system_control_bottom_right", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    ;
                                }
                                ImGui::EndChild();
                            }
                        }
                        ImGui::EndTable();
                    }
                    ImGui::SetWindowFontScale(1.0f);
                }
            }
            ImGui::EndTable();
        }

        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            stat_main = em_State::MAIN;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }

        auto [is_drag_left, mouse_delta] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
        if (is_drag_left) {
            stat_main = em_State::MAIN;
        }

        ImGui::PopID();
    }

    void show_panel_system_control()
    {
        ImGui::PushID("SYSTEM_Control");

        switch(stat_system) {
        case em_System_State::MAIN:
            show_panel_system_control_main();
            break;

        case em_System_State::FPS:
            show_panel_system_control_fps();
            break;

        case em_System_State::SHOOTING_MODE:
            show_panel_system_control_shooting_mode();
            break;

        case em_System_State::VIDEO_FORMAT:
            show_panel_system_control_video_format();
            break;

        default:
            ;
        }

        ImGui::PopID();
    }



    /////////////////////////////////////////////////////////////////
    // live view.
    /////////////////////////////////////////////////////////////////
    void show_panel_live_view(GLsizei panel_width, bool display_info = true, bool texture_simple = false)
    {
        ImGui::PushID("LIVE_VIEW");

        // if live view is OK?
        {
            proc_live_view->fetch();

            // plain image. pixel format is RGB.
            auto image_buf = proc_live_view->get_bitmap_buf();
            auto image_width = proc_live_view->get_bitmap_width();
            auto image_height = proc_live_view->get_bitmap_height();

            // update texture.
            if (image_buf != nullptr && image_width > 0 && image_height > 0) {
                if (image_width != tex_width || image_height != tex_height) {
                    tex_width = image_width;
                    tex_height = image_height;
                    tex_id = make_texture(tex_id, GL_TEXTURE_2D, tex_width, tex_height, 0, tex_type, RGB_CH_NUM);
                }
                glBindTexture(GL_TEXTURE_2D, tex_id);
                glPixelStorei(GL_PACK_ALIGNMENT, 1);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                {
                    if (tex_format == GL_RGB) tex_format = GL_BGR;
                    else if (tex_format == GL_RGBA) tex_format = GL_BGRA;
                }
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex_width, tex_height, tex_format, tex_type, image_buf);
            }

            proc_live_view->next();
        }

        auto aspect = GLfloat(tex_width) / GLfloat(tex_height);
        GLsizei w = panel_width;
        std::ostringstream sout_title;
        auto &rs = remote_server;
        std::string rs_str = rs.ip_address + ":" + rs.port + " / " + "[SRT]" + (rs.is_srt_listener ? "Listener" : "Caller") + ":" + rs.srt_port;
        sout_title << "Live View (" << rs_str << ")";
        std::string str_title = sout_title.str();
        std::vector<std::string> str_info;
        if (display_info) {
            std::ostringstream sout_info;
            sout_info << rs_str;
            str_info.push_back(sout_info.str());
        }

        if (texture_simple) {
            show_panel_texture_simple(tex_id, w, GLsizei(w / aspect), str_title.c_str(), 0, true, str_info);
        } else {
            show_panel_texture(tex_id, w, GLsizei(w / aspect), str_title.c_str(), 0, true, str_info);
        }

        ImGui::PopID();
    }

    void show_panel_live_view_with_info()
    {
        ImGui::PushID("LIVE_VIEW_WITH_INFO");

        if (ImGui::BeginChild("CHILD", ImVec2(-1, -1), ImGuiChildFlags_None, ImGuiWindowFlags_None))
        {
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImVec2 win_size = ImGui::GetWindowSize();

            show_panel_live_view(win_size.x, false, true);

            ImGui::SetWindowFontScale(0.75f);
            auto &rs = remote_server;
            std::string cgi_server = rs.ip_address + ":" + rs.port;
            std::string srt = std::string{"[SRT]"} + (rs.is_srt_listener ? "Listener" : "Caller") + ":" + rs.srt_port;
            ImGui::Text("Live View");
            ImGui::Text("%s", cgi_server.c_str());
            ImGui::Text("%s", srt.c_str());

            ImGui::SetCursorScreenPos(p);
            ImGui::InvisibleButton("##INVISIBULE_BUTTON", ImVec2(-1, win_size.x * ( 9.0f / 16.0f)));
            auto is_hovered = ImGui::IsItemHovered();
            if (is_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                stat_main_bkup = stat_main;
                stat_main = em_State::LIVE_VIEW;
            }
        }
        ImGui::EndChild();

        ImGui::PopID();
    }

    void show_panel_ptz_setting()
    {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImVec2 win_pos(viewport->WorkPos.x * vis_xscale, viewport->WorkPos.y * vis_xscale);
        ImVec2 win_size(viewport->WorkSize.x * vis_xscale, viewport->WorkSize.y * vis_xscale);

        ImGuiStyle& style = ImGui::GetStyle();
        const auto pad_frame = style.FramePadding;
        const auto pad_win = style.WindowPadding;

        set_style_color(0.0f / 7.0f);
        auto text_size = ImGui::CalcTextSize("Reset");
        auto btn_size = ImVec2(text_size.x + pad_frame.x * 2, (text_size.y + pad_frame.y) * 2);
        ImGui::SetCursorScreenPos(ImVec2(win_pos.x + (win_size.x - btn_size.x - pad_win.x), win_pos.y + (win_size.y - btn_size.y - pad_win.y)));
        if (ImGui::Button("Reset##PAN_TILT", btn_size)) {
            visca_com->send_cmd_pt_reset();
        }
        reset_style_color();
    }

    void show_panel_touch_focus_setting()
    {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImVec2 win_pos(viewport->WorkPos.x * vis_xscale, viewport->WorkPos.y * vis_xscale);
        ImVec2 win_size(viewport->WorkSize.x * vis_xscale, viewport->WorkSize.y * vis_xscale);

        ImGuiStyle& style = ImGui::GetStyle();
        const auto pad_frame = style.FramePadding;
        const auto pad_win = style.WindowPadding;

        set_style_color(2.0f / 7.0f);

        auto &ptzf = cgi->inquiry_ptzf();
        auto focus_mode = ptzf.FocusMode;
        auto text = (focus_mode == CGICmd::em_AutoManual::AUTO) ? "Auto" : "Manual";
        auto text_size = ImGui::CalcTextSize(text);
        auto btn_size = ImVec2(text_size.x + pad_frame.x * 2, (text_size.y + pad_frame.y) * 2);
        ImGui::SetCursorScreenPos(ImVec2(win_pos.x + (win_size.x - btn_size.x - pad_win.x), win_pos.y + (win_size.y - (btn_size.y + pad_win.y) * 3)));
        if (ImGui::Button(text, btn_size)) {
            auto mode = (focus_mode == CGICmd::em_AutoManual::AUTO)
                ? CGICmd::em_AutoManual::MANUAL
                : CGICmd::em_AutoManual::AUTO
                ;
            cgi->set_ptzf_FocusMode(mode);
        }

        auto touch_mode = ptzf.TouchFocusInMF;
        text = (touch_mode == CGICmd::em_TouchFocusInMF::TRACKING_AF) ? "Tracking" : "Spot";
        text_size = ImGui::CalcTextSize(text);
        btn_size = ImVec2(text_size.x + pad_frame.x * 2, (text_size.y + pad_frame.y) * 2);
        ImGui::SetCursorScreenPos(ImVec2(win_pos.x + (win_size.x - btn_size.x - pad_win.x), win_pos.y + (win_size.y - (btn_size.y + pad_win.y) * 2)));
        if (ImGui::Button(text, btn_size)) {
            auto mode = (touch_mode == CGICmd::em_TouchFocusInMF::TRACKING_AF)
                ? CGICmd::em_TouchFocusInMF::SPOT_FOCUS
                : CGICmd::em_TouchFocusInMF::TRACKING_AF
                ;
            cgi->set_ptzf_TouchFocusInMF(mode);
        }

        text_size = ImGui::CalcTextSize("Tr Cancel");
        btn_size = ImVec2(text_size.x + pad_frame.x * 2, (text_size.y + pad_frame.y) * 2);
        ImGui::SetCursorScreenPos(ImVec2(win_pos.x + (win_size.x - btn_size.x - pad_win.x), win_pos.y + (win_size.y - btn_size.y - pad_win.y)));
        if (ImGui::Button("Tr Cancel##FOCUS", btn_size)) {
            cgi->click_ptzf_FocusTrackingCancel();
        }

        reset_style_color();
    }

    void show_panel_pan_tilt()
    {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImVec2 win_size(viewport->WorkSize.x * vis_xscale, viewport->WorkSize.y * vis_xscale);

        if (is_double_click()) {
            visca_com->send_cmd_pt_home();

        } else if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            using pt_cmd = IpNetwork::VISCA_PanTilt_Command;
            auto [delta, length] = get_mouse_dragging_delta_rainbow(ImGuiMouseButton_Left);
            // auto rad = std::atan2(delta.y, delta.x);
            // auto deg = rad * 180.0f / M_PI;
            // if (deg < 0.0f) deg += 360.0f;
            // std::cout << "(delta, length) = " << delta.x << ", " << delta.y << ", " << length << " / " << deg << std::endl;
            auto lr = delta.x >= 0.0f ? pt_cmd::em_LeftRight::RIGHT : pt_cmd::em_LeftRight::LEFT;
            auto ud = delta.y >= 0.0f ? pt_cmd::em_UpDown::DOWN : pt_cmd::em_UpDown::UP;
            const auto r_max = win_size.y / 2 * 0.8f;
            const auto spd_max = pt_cmd::SPEED_MAX;
            auto pan = std::abs(std::clamp(delta.x / r_max, -1.0f, 1.0f) * spd_max);
            auto tilt = std::abs(std::clamp(delta.y / r_max, -1.0f, 1.0f) * spd_max);
            visca_com->send_cmd_pan_tilt(pan, tilt, lr, ud);

        } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            visca_com->send_cmd_pt_stop();
        }
    }

    void show_panel_zoom()
    {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImVec2 win_size(viewport->WorkSize.x * vis_xscale, viewport->WorkSize.y * vis_xscale);

        if (is_double_click()) {
            ;

        } else if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            using zm_cmd = IpNetwork::VISCA_Zoom_Command;
            auto [delta, length] = get_mouse_dragging_delta_rainbow(ImGuiMouseButton_Left);
            auto rad = std::atan2(delta.y, delta.x);
            auto deg = rad * 180.0f / M_PI;
            if (deg < 0.0f) deg += 360.0f;

            constexpr auto dt = 15;
            if (!(deg > (90 - dt) && deg < (90 + dt) || deg > (270 - dt) && deg < (270 + dt))) return;

            auto tw = delta.y >= 0.0f ? zm_cmd::em_TeleWide::WIDE : zm_cmd::em_TeleWide::TELE;
            const auto r_max = win_size.y * 0.8f;
            auto zoom = std::abs(std::clamp(delta.y / r_max, -1.0f, 1.0f) * zm_cmd::SPEED_HIGHRESO_MAX);
            visca_com->send_cmd_zoom(zoom, tw);

        } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            visca_com->send_cmd_zm_stop();
        }
    }

    void show_panel_focus()
    {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImVec2 win_size(viewport->WorkSize.x * vis_xscale, viewport->WorkSize.y * vis_xscale);

        if (is_double_click()) {
            ;

        } else if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            using focus_cmd = IpNetwork::VISCA_Focus_Command;
            auto [delta, length] = get_mouse_dragging_delta_rainbow(ImGuiMouseButton_Left);
            auto rad = std::atan2(delta.y, delta.x);
            auto deg = rad * 180.0f / M_PI;
            if (deg < 0.0f) deg += 360.0f;

            constexpr auto dt = 15;
            if (!(deg > (90 - dt) && deg < (90 + dt) || deg > (270 - dt) && deg < (270 + dt))) return;

            auto nf = delta.y >= 0.0f ? focus_cmd::em_NearFar::NEAR : focus_cmd::em_NearFar::FAR;
            const auto r_max = win_size.y * 0.8f;
            auto focus = std::abs(std::clamp(delta.y / r_max, -1.0f, 1.0f) * focus_cmd::SPEED_MAX);
            visca_com->send_cmd_focus_manual();
            visca_com->send_cmd_focus(focus, nf);

        } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            visca_com->send_cmd_focus_stop();
        }
    }

    void show_panel_touch_focus()
    {
        if (is_double_click()) {
            const ImGuiViewport* viewport = ImGui::GetMainViewport();
            ImVec2 win_size(viewport->WorkSize.x * vis_xscale, viewport->WorkSize.y * vis_xscale);

            auto &io = ImGui::GetIO();
            auto mouse_pos = io.MouseClickedPos[ImGuiMouseButton_Left];

            int x = mouse_pos.x * CGICmd::st_Ptzf::POS_X_MAX / (win_size.x - 1) + 0.5f;
            int y = mouse_pos.y * CGICmd::st_Ptzf::POS_Y_MAX / (win_size.y - 1) + 0.5f;
            cgi->set_ptzf_FocusTrackingPosition(x, y);
        }
    }



    double calc_fps() const
    {
        int fps = 30;
        double fps_d = fps;

//         auto fps_idx = Cr_Ext_get_recording_frame_rate_setting_movie(camera_handle);
//         std::string fps_str = Cr_Ext_format_recording_frame_rate_setting_movie(fps_idx);
//         if (fps_str.length() > 0) fps_str.pop_back();
//         // auto [fps_ptr, fps_ec] = std::from_chars(fps_str.begin().base(), fps_str.end().base(), fps);
// #if __has_include(<charconv>)
//         auto [fps_ptr, fps_ec] = std::from_chars(&fps_str[0], &fps_str[fps_str.length()], fps);
//         switch (fps_ec) {
//         case std::errc{}:
//             if (fps == 24 || fps == 30 || fps == 60 || fps == 120) {
//                 fps_d = fps / 1.001;
//             } else {
//                 fps_d = fps;
//             }
//             break;
//         case std::errc::invalid_argument:
// #ifndef NDEBUG
//             std::cout << "std::errc::invalid_argument : " << fps_str << std::endl;
// #endif
//             break;
//         case std::errc::result_out_of_range:
// #ifndef NDEBUG
//             std::cout << "std::errc::result_out_of_range : " << fps_str << std::endl;
// #endif
//             break;
//         default:
// #ifndef NDEBUG
//             std::cout << "std::errc ANY : " << fps_str << std::endl;
// #endif
//             break;
//         }
// #else
//     try {
//         fps = std::stoi(fps_str);
//         if (fps == 24 || fps == 30 || fps == 60 || fps == 120) {
//             fps_d = fps / 1.001;
//         } else {
//             fps_d = fps;
//         }
//     } catch(const std::exception& e) {
// #ifndef NDEBUG
//         std::cerr << e.what() << '\n';
// #endif
//     }
// #endif

        return fps_d;
    }

public:
    Gui_Window_Camera(int _width, int _height, const st_RemoteServer &_remote_server, int _queue_size)
    {
        remote_server = _remote_server;
        queue_size = _queue_size;

        tex_width = _width * vis_xscale;
        tex_height = _height * vis_xscale;
        tex_type = get_tex_type(8, false);
        std::tie(tex_internalFormat, tex_format) = get_format(tex_type, RGB_CH_NUM);
        tex_id = make_texture(0, GL_TEXTURE_2D, tex_width, tex_height, 0, tex_type, RGB_CH_NUM);

        fin_thd_blink_tally.store(true);

        // is_display_image = false;

        {
            const float st = 32768; // F1.
            const float ed = 30208; // F32.
            for (auto i = st; i > (ed - 1); i += CGICmd::iris_k_add) {
                auto fnum = CGICmd::calc_fnum(i);
                int iris = static_cast<int>(std::round(i));
                std::stringstream ss;
                ss << "F" << std::fixed << std::setprecision(2) << fnum;
                std::string fnum_str = ss.str();
                exposure_iris.emplace_back(std::pair{ iris, fnum_str });
            }
        }

        auto is_connected = CONNECT();
        camera_connection_stat.store(is_connected ? em_Camera_Connection_State::CONNECTED : em_Camera_Connection_State::DISCONNECTED);
    }

    virtual ~Gui_Window_Camera() {
        if (thd_blink_tally.joinable()) {
            thd_blink_tally.join();
        }

        if (is_CONNECTED()) DISCONNECT();

        if (tex_id > 0) glDeleteTextures(1, &tex_id);
    }

    bool display_camera_window(const std::string &win_id)
    {
        ImGuiStyle& style = ImGui::GetStyle();
        style.FrameRounding = 8.0f * vis_xscale;
        style.GrabRounding = 16.0f * vis_xscale;
        auto win_pad_bkup = style.WindowPadding;
        style.WindowPadding = ImVec2(0, 0);

        ImVec4* colors = style.Colors;
        colors[ImGuiCol_FrameBg] = ImVec4(0.43f, 0.43f, 0.43f, 0.39f);
        colors[ImGuiCol_TitleBg] = ImVec4(0.27f, 0.27f, 0.54f, 0.83f);
        colors[ImGuiCol_TitleBgActive] = ImVec4(0.32f, 0.32f, 0.63f, 0.87f);
        colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.40f, 0.40f, 0.80f, 0.20f);

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav;

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImVec2 win_pos(viewport->WorkPos.x * vis_xscale, viewport->WorkPos.y * vis_xscale);
        ImVec2 win_size(viewport->WorkSize.x * vis_xscale, viewport->WorkSize.y * vis_xscale);

#if 1
        // ImGui::SetNextWindowPos(ImVec2(0 * vis_xscale, 0 * vis_xscale), ImGuiCond_Appearing);
        // ImGui::SetNextWindowSize(ImVec2(800 * vis_xscale, 480 * vis_xscale), ImGuiCond_Appearing);
        ImGui::SetNextWindowPos(win_pos, ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(win_size, ImGuiCond_Appearing);
#else
        // Always center this window when appearing
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(800 * vis_xscale, 480 * vis_xscale), ImGuiCond_FirstUseEver);
#endif

        char str[128];
        sprintf(str, "Control##%s", win_id.c_str());
        bool is_window_opened = true;
        ImGui::Begin(str, &is_window_opened, window_flags);
        {
            switch (stat_main) {
            case em_State::MAIN:
                is_window_opened = show_panel_main();
                break;
            case em_State::SHUTTER:
                show_panel_shutter_control();
                break;
            case em_State::WHITE_BALANCE:
                show_panel_white_balance_control();
                break;
            case em_State::ISO:
                show_panel_iso_control();
                break;
            case em_State::IRIS:
                show_panel_iris_control();
                break;
            case em_State::ND:
                show_panel_nd_control();
                break;
            case em_State::SYSTEM:
                show_panel_system_control();
                break;
            default:
                ;
            }

            // Stream setting.
        }
        ImGui::End();

        style.WindowPadding = win_pad_bkup;

        return is_window_opened;
    }

    bool display_live_view(const std::string &win_id)
    {
        ImGuiStyle& style = ImGui::GetStyle();
        style.FrameRounding = 8.0f * vis_xscale;
        style.GrabRounding = 16.0f * vis_xscale;

        ImVec4* colors = style.Colors;
        colors[ImGuiCol_FrameBg] = ImVec4(0.43f, 0.43f, 0.43f, 0.39f);
        colors[ImGuiCol_TitleBg] = ImVec4(0.27f, 0.27f, 0.54f, 0.83f);
        colors[ImGuiCol_TitleBgActive] = ImVec4(0.32f, 0.32f, 0.63f, 0.87f);
        colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.40f, 0.40f, 0.80f, 0.20f);

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav;

#if 1
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImVec2 win_pos = viewport->Pos;
        ImVec2 win_size = viewport->Size;
        win_pos.x *= vis_xscale;
        win_pos.y *= vis_xscale;
        win_size.x *= vis_xscale;
        win_size.y *= vis_xscale;

        // ImGui::SetNextWindowPos(ImVec2(0 * vis_xscale, 0 * vis_xscale), ImGuiCond_Appearing);
        // ImGui::SetNextWindowSize(ImVec2(800 * vis_xscale, 480 * vis_xscale), ImGuiCond_Appearing);
        ImGui::SetNextWindowPos(win_pos, ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(win_size, ImGuiCond_Appearing);
#else
        // Always center this window when appearing
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
#endif

        char str[128];
        sprintf(str, "Live View##%s", win_id.c_str());
        bool is_window_opened = true;
        ImGui::Begin(str, &is_window_opened, window_flags);
        {
            show_panel_live_view(tex_width);

            {
                ImGui::SetCursorScreenPos(win_pos);
                if (ImGui::BeginChild("##Function", ImVec2(0, 0), false, window_flags)) {
                    ImGuiStyle& style = ImGui::GetStyle();
                    const auto pad_frame = style.FramePadding;
                    const auto pad_win = style.WindowPadding;
                    auto text_size = ImGui::CalcTextSize("@T@");
                    ImVec2 btn_size(text_size.x + pad_frame.x * 2, (text_size.y + pad_frame.y) * 2);
                    ImGui::SetCursorScreenPos(ImVec2(win_pos.x + (win_size.x - btn_size.x - pad_win.x), win_pos.y));

                    // show_panel_blink_tally(false, true);
                    show_panel_blink_tally(remote_server.ip_address);

                    // select PTZ, FOCUS.
                    {
                        set_style_color(5.0f / 7.0f);

                        auto text = (stat_ptzf == em_Ptzf_State::PTZ) ? "Focus" : "PTZ";
                        auto text_size = ImGui::CalcTextSize(text);
                        ImVec2 btn_size(text_size.x + pad_frame.x * 2, (text_size.y + pad_frame.y) * 2);
                        ImGui::SetCursorScreenPos(ImVec2(win_pos.x, win_pos.y + (win_size.y - btn_size.y - pad_win.y)));
                        if (ImGui::Button(text, btn_size)) {
                            if (stat_ptzf == em_Ptzf_State::PTZ) {
                                stat_ptzf = em_Ptzf_State::FOCUS;
                            } else {
                                stat_ptzf = em_Ptzf_State::PTZ;
                            }
                        }

                        reset_style_color();
                    }

                    switch(stat_ptzf) {
                    case em_Ptzf_State::PTZ:
                        show_panel_ptz_setting();
                        break;

                    case em_Ptzf_State::FOCUS:
                        show_panel_touch_focus_setting();
                        break;

                    default:
                        ;
                    }
               }
                ImGui::EndChild();
            }

            if (stat_ptzf == em_Ptzf_State::PTZ) {
                auto &io = ImGui::GetIO();
                auto mouse_pos = io.MouseClickedPos[ImGuiMouseButton_Left];
                auto center = ImGui::GetMainViewport()->GetCenter();

                if (std::abs(mouse_pos.x - center.x) < win_size.x / 10
                    && std::abs(mouse_pos.y - center.y) < win_size.x / 10
                ) {
                    show_panel_pan_tilt();
                }
                if (mouse_pos.x < win_size.x / 3) show_panel_zoom();
                if (mouse_pos.x > win_size.x / 3 * 2) show_panel_focus();

            } else if (stat_ptzf == em_Ptzf_State::FOCUS) {
                auto &io = ImGui::GetIO();
                auto mouse_pos = io.MouseClickedPos[ImGuiMouseButton_Left];

                if (mouse_pos.x < win_size.x / 16) show_panel_focus();

                show_panel_touch_focus();
            }

            if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                stat_main = stat_main_bkup;
            }

            // auto is_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
            // if (is_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            //     stat_main = stat_main_bkup;
            // }
            auto [is_drag_left, mouse_delta] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
            if (is_drag_left
                && std::abs(mouse_delta.x) > win_size.x * 0.8f
                && std::abs(mouse_delta.y) < win_size.y * 0.2f
            ) {
                stat_main = stat_main_bkup;
            }
        }
        ImGui::End();

        return is_window_opened;
    }

    bool display_http_digest_login_window(const std::string &win_id)
    {
        ImGuiStyle& style = ImGui::GetStyle();
        style.FrameRounding = 8.0f * vis_xscale;
        style.GrabRounding = 16.0f * vis_xscale;

        ImVec4* colors = style.Colors;
        colors[ImGuiCol_FrameBg] = ImVec4(0.43f, 0.43f, 0.43f, 0.39f);
        colors[ImGuiCol_TitleBg] = ImVec4(0.27f, 0.27f, 0.54f, 0.83f);
        colors[ImGuiCol_TitleBgActive] = ImVec4(0.32f, 0.32f, 0.63f, 0.87f);
        colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.40f, 0.40f, 0.80f, 0.20f);

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
        window_flags = window_flags & ~ImGuiWindowFlags_NoTitleBar;

#if 0
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImVec2 win_pos(viewport->WorkPos.x * vis_xscale, viewport->WorkPos.y * vis_xscale);
        ImVec2 win_size(viewport->WorkSize.x * vis_xscale, viewport->WorkSize.y * vis_xscale);

        // ImGui::SetNextWindowPos(ImVec2(0 * vis_xscale, 0 * vis_xscale), ImGuiCond_Appearing);
        // ImGui::SetNextWindowSize(ImVec2(800 * vis_xscale, 480 * vis_xscale), ImGuiCond_Appearing);
        ImGui::SetNextWindowPos(win_pos, ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(win_size, ImGuiCond_Appearing);
#else
        // Always center this window when appearing
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_Appearing);
#endif

        bool is_window_opened = true;

        bool display_keyboard = display_input_username || display_input_password;
        if (!display_keyboard) {
            if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                camera_connection_stat.store(em_Camera_Connection_State::NO_AUTH);
                is_window_opened = false;
            }

            auto [is_drag_left, mouse_delta] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
            if (is_drag_left) {
                camera_connection_stat.store(em_Camera_Connection_State::NO_AUTH);
                is_window_opened = false;
            }
        }

        // char str[128];
        // sprintf(str, "Login -->##%s", win_id.c_str());
        auto &rs = remote_server;
        std::string str = rs.ip_address + ":" + rs.port + " / " + "[SRT]" + (rs.is_srt_listener ? "Listener" : "Caller") + ":" + rs.srt_port;
        ImGui::Begin(str.c_str(), &is_window_opened, window_flags);
        {
            // // print selected camera info.
            // show_panel_print_selected_camera_info();

            auto &rs = remote_server;
            // user name.
            {
                auto str = "user name: ";
                ImGui::Text("%s", str); ImGui::SameLine();
                auto req = show_panel_inputtext(str, rs.username, 200);
                if (req) display_input_username = true;
                if (display_input_username) {
                    display_input_username = kb.display_keyboard_window(req, str, rs.username);
                }
            }

            // password.
            {
                auto str = "password:  ";
                ImGui::Text("%s", str); ImGui::SameLine();
                auto req = show_panel_inputtext(str, rs.password, 200, true);
                if (req) display_input_password = true;
                if (display_input_password) {
                    display_input_password = kb.display_keyboard_window(req, str, rs.password, Gui_Window_Keyboard::em_KeyboardPattern::LOWER, true);
                }
            }

            if (ImGui::Button("Login")) {
                cgi->set_account(remote_server.username, remote_server.password);
                camera_connection_stat.store(em_Camera_Connection_State::CONNECTED);
                auto col = (gui_skin != em_GuiSkin::LIGHT) ? ImVec4{1, 1, 0, 1} : ImVec4{0.7f, 0.7f, 0, 1};
                ImGui::SameLine(); ImGui::TextColored(col, "Now trying login..... Please wait");
            }
        }
        ImGui::End();

        return is_window_opened;
    }

    bool display_control_window(const std::string &win_id)
    {
        bool is_window_opened = true;

        switch (camera_connection_stat.load()) {
        case em_Camera_Connection_State::DISCONNECTED:
            is_window_opened = false;
            break;

        case em_Camera_Connection_State::NO_AUTH:
            is_window_opened = display_http_digest_login_window(win_id);
            break;

        case em_Camera_Connection_State::CONNECTED:
            // check_auth();
            if (!cgi->is_auth()) {
                camera_connection_stat.store(em_Camera_Connection_State::NO_AUTH);
                break;
            }

            // Run Live View thread.
            if (!thd_proc_live_view.joinable()) {
                {
                    // SRT-Listener ?
                    CGICmd::st_Stream stream = {};
                    cgi->inquiry(stream);
                    if (cgi->is_timeout() || !cgi->is_connected()) {
                        is_window_opened = false;
                        break;
                    }
                    if (stream.StreamMode != CGICmd::StreamMode_SRT_LISTENER) {
                        stream_mode_bkup = stream.StreamMode;
                        cgi->set_stream_StreamMode(CGICmd::StreamMode_SRT_LISTENER);
                        constexpr auto RETRY_MAX = 60;
                        int retry_count = RETRY_MAX;
                        do {
                            cgi->inquiry(stream);
                            retry_count--;
                        } while (retry_count > 0 && stream.StreamMode != CGICmd::StreamMode_SRT_LISTENER);
                        if (retry_count == 0) {
                            is_window_opened = false;
                            break;
                        }
                    }

                    // get SRT port #.
                    CGICmd::st_Srt srt = {};
                    cgi->inquiry(srt);
                    remote_server.srt_port = std::to_string(srt.SrtListenPort);

                    proc_live_view.reset(new ProcLiveView(remote_server, tex_width, tex_height, queue_size));
                    if (proc_live_view && proc_live_view->is_running()) {
                        std::thread thd_tmp{ [&]{ proc_live_view->run(); }};
                        thd_proc_live_view = std::move(thd_tmp);
                    } else {
                        proc_live_view.reset();
                    }
                }
            }

            switch (stat_main) {
            case em_State::MAIN:
            case em_State::SHUTTER:
            case em_State::WHITE_BALANCE:
            case em_State::ISO:
            case em_State::IRIS:
            case em_State::ND:
            case em_State::SYSTEM:
                {
                    auto is_update = cgi->is_update_cmd_info();

                    if (is_update) cgi->fetch();

                    is_window_opened = display_camera_window(win_id);

                    if (is_update) cgi->next();
                }
                break;

            case em_State::LIVE_VIEW:
                {
                    auto is_update = cgi->is_update_cmd_info();

                    if (is_update) cgi->fetch();

                    is_window_opened = display_live_view(win_id);

                    if (is_update) cgi->next();
                }
                break;
            default:
                ;
            }

            break;

        default:
            is_window_opened = false;
        }

        return is_window_opened;
    }

    auto get_camera_connection_stat() const { return camera_connection_stat.load(); }
    bool is_DISCONNECTED() const { return camera_connection_stat.load() == em_Camera_Connection_State::DISCONNECTED; }
    bool is_NO_AUTH() const { return camera_connection_stat.load() == em_Camera_Connection_State::NO_AUTH; }
    bool is_CONNECTED() const { return camera_connection_stat.load() == em_Camera_Connection_State::CONNECTED; }

    double get_fps() const { return calc_fps(); }

    bool CONNECT()
    {
        if (is_CONNECTED()) return false;

        bool ret_inq = false;
        bool ret_set = false;

        if (!thd_cgi_inq.joinable()) {
            cgi = std::make_unique<CGI>(remote_server.ip_address, stoi(remote_server.port), remote_server.username, remote_server.password);
            if (cgi && cgi->is_running()) {
                std::thread thd_tmp{ [&]{ cgi->run_inq(); }};
                thd_cgi_inq = std::move(thd_tmp);
                ret_inq = true;
            } else {
                ret_inq = false;
            }
        }
        if (!thd_cgi_set.joinable()) {
            if (cgi && cgi->is_running()) {
                std::thread thd_tmp{ [&]{ cgi->run_set(); }};
                thd_cgi_set = std::move(thd_tmp);
                ret_set = true;
            } else {
                ret_set = false;
            }
        }

        visca_com = IpNetwork::VISCA_Com::Create(remote_server.ip_address);
        bool ret_visca = visca_com ? true : false;

        auto ret = (ret_inq && ret_set && ret_visca);

        if (ret) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));   // TODO: wait event.
        }

        return ret;
    }

    bool DISCONNECT()
    {
        cgi->set_stream_StreamMode(stream_mode_bkup);

        if (proc_live_view) {
            if (proc_live_view->is_running()) proc_live_view->stop();
            if (thd_proc_live_view.joinable()) thd_proc_live_view.join();
        }

        if (cgi) {
            if (cgi->is_running()) cgi->stop();
            if (thd_cgi_inq.joinable()) thd_cgi_inq.join();
            if (thd_cgi_set.joinable()) thd_cgi_set.join();
        }

        return true;
    }

    void sync_remote_server(st_RemoteServer &rs)
    {
        rs = remote_server;
    }
};
