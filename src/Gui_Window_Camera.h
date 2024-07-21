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
        FPS,

        PTZ,
        FOCUS,
        STREAMING,

        LIVE_VIEW,
    };

    enum class em_StateShutter {
        MODE,
        CONTROL,
    };

    enum class em_StateWhiteBalance {};
    enum class em_StateISO {};
    enum class em_StateIRIS {};
    enum class em_StateND {};
    enum class em_StateFPS {};

private:
    const std::list<std::pair<CGICmd::em_ExposureShutterModeState, std::string>> exposure_shutter_mode_state = {
        {CGICmd::ExposureShutterModeState_AUTO, "Auto"},
        {CGICmd::ExposureShutterModeState_SPEED, "Speed"},
        {CGICmd::ExposureShutterModeState_ANGLE, "Angle"},
        {CGICmd::ExposureShutterModeState_ECS, "ECS"},
        {CGICmd::ExposureShutterModeState_OFF, "Off"},
    };

    st_RemoteServer remote_server = {};

    em_State stat_main = em_State::MAIN;
    em_State stat_main_bkup = em_State::MAIN;
    em_StateShutter stat_shutter = em_StateShutter::CONTROL;
    CGICmd::em_ExposureShutterModeState shutter_mode_state_bkup = CGICmd::ExposureShutterModeState_AUTO;

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

    std::unique_ptr<CGI> cgi;
    std::thread thd_cgi_inq;
    std::thread thd_cgi_set;

    StopWatch sw;

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
        if (ImGui::Button(str.c_str())) {
            ;
        }

        ImGui::SameLine();

        auto color = ImVec4{0, 1, 0, 1};
        auto status_str = "Not Recording";
        ImGui::TextColored(color, "%s", status_str);

        reset_style_color();

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
                const auto text_size = ImGui::CalcTextSize("A");
                const auto pad_frame = style.FramePadding;
                const ImVec2 ar_size(text_size.x + pad_frame.x, text_size.y + pad_frame.y);
                ImVec2 p_up(p.x + (win_size.x / 2) - (ar_size.x / 2) - pad_frame.x, p.y + 4.0f);
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

                    for (auto &[k, v]: vec) {
                        if (k == idx) {
                            set_style_color(4.0f / 7.0f, 0.9f, 0.9f);

                            ImGui::Button(v.c_str(), ImVec2(-1, 0));
                            ImGui::SetScrollHereY(0.5f); // 0.0f:top, 0.5f:center, 1.0f:bottom

                            reset_style_color();

                        } else {
                            style.FrameBorderSize = 0.0f;
                            set_style_color(5.0f, 0.1f, 0.1f);

                            ImGui::Button(v.c_str(), ImVec2(-1, 0));

                            reset_style_color();
                            style.FrameBorderSize = fb;
                        }
                    }

                    style.FrameRounding = fr;
                    style.GrabRounding = gr;
                }
                ImGui::EndChild();

                // centering.
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



    /////////////////////////////////////////////////////////////////
    // Main control panel.
    /////////////////////////////////////////////////////////////////
    void show_panel_main()
    {
        ImGui::PushID("Main");

        stat_main_bkup = stat_main;

        auto tbl_flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        auto cld_flags = ImGuiChildFlags_None;
        auto win_flags = ImGuiWindowFlags_None;

        auto &imaging = cgi->inquiry_imaging();

        if (ImGui::BeginTable("main", 1, tbl_flags))
        {
            auto p = ImGui::GetCursorScreenPos();
            auto sz = ImGui::GetWindowSize();
            float min_row_height = (sz.y - p.y - 10) / 3;

            for (int row = 0; row < 3; row++)
            {
                ImGui::TableNextRow(ImGuiTableRowFlags_None, min_row_height);
                ImGui::TableNextColumn();

                if (row == 0) {
                    if (ImGui::BeginTable("main_top", 3, ImGuiTableFlags_BordersInnerV))
                    {
                        for (int row = 0; row < 1; row++)
                        {
                            ImGui::TableNextRow();

                            ImGui::TableSetColumnIndex(0);
                            {
                                if (ImGui::BeginChild("main_top_left", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    if (ImGui::Button("FPS") || ImGui::IsKeyPressed(ImGuiKey_W, false)) {
                                        stat_main = em_State::FPS;
                                    }
                                }
                                ImGui::EndChild();
                            }

                            ImGui::TableSetColumnIndex(1);
                            {
                                if (ImGui::BeginChild("main_top_center", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    if (ImGui::Button("ISO") || ImGui::IsKeyPressed(ImGuiKey_E, false)) {
                                        stat_main = em_State::ISO;
                                    }
                                }
                                ImGui::EndChild();
                            }

                            ImGui::TableSetColumnIndex(2);
                            {
                                if (ImGui::BeginChild("main_top_right", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    if (ImGui::Button("Shutter") || ImGui::IsKeyPressed(ImGuiKey_R, false)) {
                                        stat_main = em_State::SHUTTER;
                                    }
                                    auto idx = imaging.ExposureShutterModeState;
                                    auto &vec = exposure_shutter_mode_state;
                                    auto itr = std::find_if(vec.begin(), vec.end(), [&idx](auto &e){ return e.first == idx; });
                                    std::string mode_str = "---";
                                    if (itr != vec.end()) {
                                        auto &[k, v] = *itr;
                                        mode_str = v;
                                    }
                                    ImGui::Text("%s", mode_str.c_str());
                                    show_panel_shutter_value();
                                }
                                ImGui::EndChild();
                            }
                        }
                        ImGui::EndTable();
                    }

                } else if (row == 1) {
                    if (ImGui::BeginTable("main_middle", 2, ImGuiTableFlags_BordersInnerV))
                    {
                        for (int row = 0; row < 1; row++)
                        {
                            ImGui::TableNextRow();

                            ImGui::TableSetColumnIndex(0);
                            {
                                ImGui::SetWindowFontScale(0.75f);

                                // print selected camera info.
                                show_panel_print_selected_camera_info();

                                // print FPS.
                                show_panel_print_fps();

                                ImGui::SetWindowFontScale(1.0f);

                                // movie rec.
                                show_panel_movie_rec();
                            }

                            ImGui::TableSetColumnIndex(1);
                            {
                                ImVec2 p = ImGui::GetCursorScreenPos();
                                ImVec2 win_size = ImGui::GetWindowSize();

                                show_panel_live_view(win_size.x / 4, false, true);
                            }
                        }
                        ImGui::EndTable();
                    }

                } else if (row == 2) {
                    if (ImGui::BeginTable("main_bottom", 3, ImGuiTableFlags_BordersInnerV))
                    {
                        for (int row = 0; row < 1; row++)
                        {
                            ImGui::TableNextRow();

                            ImGui::TableSetColumnIndex(0);
                            {
                                if (ImGui::BeginChild("main_bottom_left", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    if (ImGui::Button("ND") || ImGui::IsKeyPressed(ImGuiKey_X, false)) {
                                        stat_main = em_State::ND;
                                    }
                                }
                                ImGui::EndChild();
                            }

                            ImGui::TableSetColumnIndex(1);
                            {
                                if (ImGui::BeginChild("main_bottom_center", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    if (ImGui::Button("IRIS") || ImGui::IsKeyPressed(ImGuiKey_C, false)) {
                                        stat_main = em_State::IRIS;
                                    }
                                }
                                ImGui::EndChild();
                            }

                            ImGui::TableSetColumnIndex(2);
                            {
                                if (ImGui::BeginChild("main_bottom_right", ImVec2(-1, min_row_height), cld_flags, win_flags)) {
                                    if (ImGui::Button("WB") || ImGui::IsKeyPressed(ImGuiKey_V, false)) {
                                        stat_main = em_State::WHITE_BALANCE;
                                    }
                                }
                                ImGui::EndChild();
                            }
                        }
                        ImGui::EndTable();
                    }
                }
            }
            ImGui::EndTable();
        }

        ImGui::PopID();
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
    // F number control.
    /////////////////////////////////////////////////////////////////
    void show_panel_f_number_control()
    {
        ImGui::PushID("F_Number_Control");

        if (ImGui::Button("F#")) {
            ;
        }

        ImGui::PopID();
    }



    /////////////////////////////////////////////////////////////////
    // iso sensitivity control.
    /////////////////////////////////////////////////////////////////
    void show_panel_iso_sensitivity_control()
    {
        ImGui::PushID("ISO_Sensitivity_Control");

        if (ImGui::Button("ISO")) {
            ;
        }

        ImGui::PopID();
    }



    /////////////////////////////////////////////////////////////////
    // shutter control.
    /////////////////////////////////////////////////////////////////
    void show_panel_shutter_value()
    {
        auto &imaging = cgi->inquiry_imaging();
        auto &state = imaging.ExposureShutterModeState;

        switch (state) {
        case CGICmd::ExposureShutterModeState_SPEED:
        case CGICmd::ExposureShutterModeState_AUTO:
            {
                auto &project = cgi->inquiry_project();
                auto frame_rate = project.RecFormatFrequency;
                auto idx = imaging.ExposureExposureTime;
                auto idx_min = imaging.ExposureExposureTimeRange.min;
                auto idx_max = imaging.ExposureExposureTimeRange.max;

                std::string str = "---";
                if (CGICmd::exposure_exposure_time.contains(frame_rate)) {
                    auto &lst = CGICmd::exposure_exposure_time[frame_rate];
                    using e_type = decltype(lst.front());
                    auto itr = std::find_if(lst.begin(), lst.end(), [&idx](e_type e){ return e.first == idx; });
                    if (itr != lst.end()) {
                        str = (*itr).second;
                    }
                }
                ImGui::Text("%s", str.c_str());
            }
            break;

        case CGICmd::ExposureShutterModeState_ANGLE:
            {
                auto idx = imaging.ExposureAngle;
                auto idx_min = imaging.ExposureAngleRange.min;
                auto idx_max = imaging.ExposureAngleRange.max;

                std::string str = "---";
                auto &lst = CGICmd::exposure_angle;
                using e_type = decltype(lst.front());
                auto itr = std::find_if(lst.begin(), lst.end(), [&idx](e_type e){ return e.first == idx; });
                if (itr != lst.end()) {
                    str = (*itr).second;
                }
                ImGui::Text("%s", str.c_str());
            }
            break;

        case CGICmd::ExposureShutterModeState_ECS:
            {
                auto idx = imaging.ExposureECS;
                auto val = imaging.ExposureECSValue;
                auto val_f = val / 1000.0f;
                ImGui::Text("[%d] %.2f", idx, val_f);
            }
            break;

        case CGICmd::ExposureShutterModeState_OFF:
        default:
            ;
        }
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
            if (state == CGICmd::ExposureShutterModeState_AUTO || state == CGICmd::ExposureShutterModeState_OFF) {
                stat_main = em_State::MAIN;
            } else {
                stat_shutter = em_StateShutter::CONTROL;
            }
        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
            stat_main_bkup = stat_main;
            stat_main = em_State::LIVE_VIEW;
        }

        ImGui::PopID();
    }

    void show_panel_shutter_speed()
    {
        ImGui::PushID("SHUTTER_SPEED");

        auto &imaging = cgi->inquiry_imaging();
        auto &state = imaging.ExposureShutterModeState;

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("shutter_speed", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);

                ImGui::TableSetColumnIndex(1);
                {
                    auto &project = cgi->inquiry_project();
                    auto frame_rate = project.RecFormatFrequency;
                    auto &lst = (CGICmd::exposure_exposure_time.contains(frame_rate))
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
        auto &state = imaging.ExposureShutterModeState;

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("shutter_angle", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);

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
        auto &state = imaging.ExposureShutterModeState;

        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("shutter_ecs", 3, flags))
        {
            for (int row = 0; row < 1; row++)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);

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
                stat_shutter = em_StateShutter::MODE;
                break;

            case CGICmd::ExposureShutterModeState_OFF:
                stat_shutter = em_StateShutter::MODE;
                break;

            default:
                stat_main = em_State::MAIN;
            }

            // show_panel_shutter_value();

            if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                stat_main = em_State::MAIN;
            } else if (ImGui::IsKeyPressed(ImGuiKey_X, false)) {
                stat_shutter = em_StateShutter::MODE;
            } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
                stat_main_bkup = stat_main;
                stat_main = em_State::LIVE_VIEW;
            }
        }

        ImGui::PopID();
    }



    /////////////////////////////////////////////////////////////////
    // white_balance control.
    /////////////////////////////////////////////////////////////////
    void show_panel_white_balance_control()
    {
        ImGui::PushID("White_Balance_Control");

        if (ImGui::Button("White Balance")) {
            ;
        }

        ImGui::PopID();
    }



    /////////////////////////////////////////////////////////////////
    // live view.
    /////////////////////////////////////////////////////////////////
    void show_panel_live_view(GLsizei panel_width, bool display_info = true, bool texture_simple = false)
    {
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
    }

    void show_panel_live_view_with_info()
    {
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
        }
        ImGui::EndChild();
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
    Gui_Window_Camera(int _width, int _height, const st_RemoteServer &_remote_server)
    {
        remote_server = _remote_server;

        tex_width = _width * vis_xscale;
        tex_height = _height * vis_xscale;
        tex_type = get_tex_type(8, false);
        std::tie(tex_internalFormat, tex_format) = get_format(tex_type, RGB_CH_NUM);
        tex_id = make_texture(0, GL_TEXTURE_2D, tex_width, tex_height, 0, tex_type, RGB_CH_NUM);

        // is_display_image = false;

        auto is_connected = CONNECT();
        camera_connection_stat.store(is_connected ? em_Camera_Connection_State::CONNECTED : em_Camera_Connection_State::DISCONNECTED);
    }

    virtual ~Gui_Window_Camera() {
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
                {
                    show_panel_main();

                    if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
                        stat_main = em_State::LIVE_VIEW;
                    } else if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                        is_window_opened = false;
                    }
                }
                break;
            case em_State::SHUTTER:
                show_panel_shutter_control();
                break;
            case em_State::WHITE_BALANCE:
                break;
            case em_State::ISO:
                break;
            case em_State::IRIS:
                break;
            case em_State::ND:
                break;
            case em_State::FPS:
                break;
            default:
                ;
            }

            // PTZ
            // focus.
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

            if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
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

        // char str[128];
        // sprintf(str, "Login -->##%s", win_id.c_str());
        auto &rs = remote_server;
        std::string str = rs.ip_address + ":" + rs.port + " / " + "[SRT]" + (rs.is_srt_listener ? "Listener" : "Caller") + ":" + rs.srt_port;
        bool is_window_opened = true;
        ImGui::Begin(str.c_str(), &is_window_opened, window_flags);
        {
            // // print selected camera info.
            // show_panel_print_selected_camera_info();

            auto &rs = remote_server;
            // user name.
            {
                auto str = "user name: ";
                ImGui::Text("%s", str); ImGui::SameLine();
                show_panel_inputtext(str, rs.username, 200);
            }

            // password.
            {
                auto str = "password:  ";
                ImGui::Text("%s", str); ImGui::SameLine();
                show_panel_inputtext(str, rs.password, 200, true);
            }

            if (ImGui::Button("Login")) {
                cgi->set_account(remote_server.username, remote_server.password);
                camera_connection_stat.store(em_Camera_Connection_State::CONNECTED);
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
            {
                if (!cgi->is_auth()) {
                    camera_connection_stat.store(em_Camera_Connection_State::NO_AUTH);
                    break;
                }
            }

            switch (stat_main) {
            case em_State::MAIN:
            case em_State::SHUTTER:
            case em_State::WHITE_BALANCE:
            case em_State::ISO:
            case em_State::IRIS:
            case em_State::ND:
            case em_State::FPS:
                {
                    auto is_update = cgi->is_update_cmd_info();

                    if (is_update) cgi->fetch();

                    is_window_opened = display_camera_window(win_id);

                    if (is_update) cgi->next();
                }
                break;

            case em_State::LIVE_VIEW:
                {
                    is_window_opened = display_live_view(win_id);
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

        bool ret = false;
        bool ret_inq = false;
        bool ret_set = false;

        if (!thd_proc_live_view.joinable()) {
            proc_live_view.reset(new ProcLiveView(remote_server, tex_width, tex_height));
            if (proc_live_view && proc_live_view->is_running()) {
                std::thread thd_tmp{ [&]{ proc_live_view->run(); }};
                thd_proc_live_view = std::move(thd_tmp);
                ret = true;
            }
        }

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

        ret = (ret && ret_inq && ret_set);

        if (ret) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));   // TODO: wait event.
        }

        return ret;
    }

    bool DISCONNECT()
    {
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

};
