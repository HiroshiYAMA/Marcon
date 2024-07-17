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
private:
    st_RemoteServer remote_server = {};

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
    bool is_display_image;

    std::unique_ptr<CGI> cgi;
    std::thread thd_cgi_inq;
    std::thread thd_cgi_set;

    StopWatch sw;

    // template<typename F_get, typename F_set, typename F_get_list, typename F_format, typename F_format_release, typename T>
    // void put_item_change_property(
    //     F_get func_get,
    //     F_set func_set,
    //     F_get_list func_get_list,
    //     F_format func_format,
    //     F_format_release func_format_release,
    //     T &val_list,
    //     const char *name,
    //     const char *item_id,
    //     size_t str_len_max,
    //     bool same_line = true
    // )
    // {
    //     ImGui::BeginGroup();

    //     auto val = func_get(camera_handle);
    //     using vec_type = std::pair<decltype(val), std::string>;

    //     auto num = func_get_list(camera_handle, &val_list);
    //     std::vector<vec_type> vec;
    //     make_list2vec_with_free(val_list, vec, func_format, func_format_release);

    //     int idx = search_vec(vec, val);

    //     // repair list.
    //     repair_list_with_free(vec, idx, val, func_format, func_format_release);

    //     std::vector<char> combo_vec;
    //     make_combo_vec(vec, combo_vec, str_len_max);

    //     ImGui::Text("%s", name);
    //     if (same_line) ImGui::SameLine();
    //     ImGui::PushItemWidth(str_len_max * vis_xscale);
    //     ImGui::Combo(item_id, &idx, combo_vec.data());
    //     ImGui::PopItemWidth();

    //     if (idx >= 0 && idx < vec.size()) {
    //         auto val_new = vec[idx].first;
    //         if (val != val_new) func_set(camera_handle, val_new);
    //     }

    //     ImGui::EndGroup();
    // }
    // template<typename F_get, typename F_get_list, typename F_format, typename F_format_release, typename T>
    // void set_metadata_camera_status_list(
    //     F_get func_get,
    //     F_get_list func_get_list,
    //     F_format func_format,
    //     F_format_release func_format_release,
    //     T &val_list,
    //     st_Metadata_CameraStatus_list &camera_status_list
    // )
    // {
    //     auto str = func_format(func_get(camera_handle));
    //     camera_status_list.value = str;
    //     func_format_release(str);

    //     auto num = func_get_list(camera_handle, &val_list);
    //     auto &list = camera_status_list.list;
    //     list.clear();
    //     for (int i = 0; i < num; i++) {
    //         auto str = func_format(val_list.list[i]);
    //         list.push_back(str);
    //         func_format_release(str);
    //     }
    // }
    // template<typename F_get, typename F_get_list, typename F_format, typename F_format_release, typename T>
    // void set_metadata_camera_status_range(
    //     F_get func_get,
    //     F_get_list func_get_list,
    //     F_format func_format,
    //     F_format_release func_format_release,
    //     T &val_list,
    //     st_Metadata_CameraStatus_range &camera_status_range
    // )
    // {
    //     auto str = func_format(func_get(camera_handle));
    //     camera_status_range.value = str;
    //     func_format_release(str);

    //     auto num = func_get_list(camera_handle, &val_list);
    //     if (num == 3) {
    //         auto &range = camera_status_range.range;
    //         auto &vlist = val_list.list;
    //         range = { vlist[0], vlist[1], vlist[2] };
    //     }
    // }
    // static constexpr void dummy_func_format_release(const char *val) {}

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

    // focus control.
    void show_panel_focus_control()
    {
        ImGui::PushID("Focus_Control");

        if (ImGui::Button("Focus")) {
            ;
        }

        ImGui::PopID();
    }

    // F number control.
    void show_panel_f_number_control()
    {
        ImGui::PushID("F_Number_Control");

        if (ImGui::Button("F#")) {
            ;
        }

        ImGui::PopID();
    }

    // iso sensitivity control.
    void show_panel_iso_sensitivity_control()
    {
        ImGui::PushID("ISO_Sensitivity_Control");

        if (ImGui::Button("ISO")) {
            ;
        }

        ImGui::PopID();
    }

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

            float child_w = (ImGui::GetContentRegionAvail().x - 4 * style.ItemSpacing.x) / 3;
            float child_h = (ImGui::GetContentRegionAvail().y - 4 * style.ItemSpacing.y);
            if (child_w < 1.0f) child_w = 1.0f;
            if (child_h < 1.0f) child_h = 1.0f;
            if (ImGui::BeginChild("CHILD", ImVec2(child_w, 200.0f), ImGuiChildFlags_None, ImGuiWindowFlags_None))
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
                p.x += (win_size.x / 2) - 20.0f;
                ImGui::SetCursorScreenPos(p);
                ImGui::PushButtonRepeat(true);
                if (ImGui::ArrowButton("##UP", ImGuiDir_Up) || ImGui::IsKeyPressed(ImGuiKey_E)) {
                    if (itr != vec.begin()) itr--;
                    is_changed = true;
                }
                ImGui::PopButtonRepeat();

                if (ImGui::BeginChild("CHILD_CHILD", ImVec2(-1, 140.0f), ImGuiChildFlags_Border, ImGuiWindowFlags_None))
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
                auto pp = ImGui::GetCursorScreenPos();
                pp.x = p.x;
                ImGui::SetCursorScreenPos(pp);
                ImGui::PushButtonRepeat(true);
                if (ImGui::ArrowButton("##DOWN", ImGuiDir_Down) || ImGui::IsKeyPressed(ImGuiKey_C)) {
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

    // shutter control.
    void show_panel_shutter_control()
    {
        ImGui::PushID("Shutter_Control");

        auto &imaging = cgi->inquiry_imaging();
        auto &state = imaging.ExposureShutterModeState;

        // select shutter mode.
        {
            static const std::list<std::pair<CGICmd::em_ExposureShutterModeState, std::string>> exposure_shutter_mode_state = {
                {CGICmd::ExposureShutterModeState_AUTO, "Auto"},
                {CGICmd::ExposureShutterModeState_SPEED, "Speed"},
                {CGICmd::ExposureShutterModeState_ANGLE, "Angle"},
                {CGICmd::ExposureShutterModeState_ECS, "ECS"},
                {CGICmd::ExposureShutterModeState_OFF, "Off"},
            };

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

        // shutter speed.
        if (state == CGICmd::ExposureShutterModeState_SPEED)
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

        // shutter angle.
        if (state == CGICmd::ExposureShutterModeState_ANGLE)
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

        // shutter ECS.
        if (state == CGICmd::ExposureShutterModeState_ECS)
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

        // display shutter value.
        {
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

        if (ImGui::Button("Shutter")) {
            ;
        }

        ImGui::PopID();
    }

    // white_balance control.
    void show_panel_white_balance_control()
    {
        ImGui::PushID("White_Balance_Control");

        if (ImGui::Button("White Balance")) {
            ;
        }

        ImGui::PopID();
    }

    bool show_panel_live_view()
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
        GLsizei w = tex_width;
        std::ostringstream sout_title;
        auto &rs = remote_server;
        std::string rs_str = rs.ip_address + ":" + rs.port + " / " + "[SRT]" + (rs.is_srt_listener ? "Listener" : "Caller") + ":" + rs.srt_port;
        sout_title << "Live View (" << rs_str << ")";
        std::string str_title = sout_title.str();
        std::vector<std::string> str_info;
        {
            std::ostringstream sout_info;
            sout_info << rs_str;
            str_info.push_back(sout_info.str());
        }
        // show_panel_texture(tex_id, w, GLsizei(w / aspect), str_title.c_str(), 0, true, str_info);
        bool is_opend = true;
        display_texture(tex_id, w, GLsizei(w / aspect), str_title.c_str(), &is_opend, 0, false, false, true, str_info);

        return is_opend;
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

        is_display_image = false;

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

        ImVec4* colors = style.Colors;
        colors[ImGuiCol_FrameBg] = ImVec4(0.43f, 0.43f, 0.43f, 0.39f);
        colors[ImGuiCol_TitleBg] = ImVec4(0.27f, 0.27f, 0.54f, 0.83f);
        colors[ImGuiCol_TitleBgActive] = ImVec4(0.32f, 0.32f, 0.63f, 0.87f);
        colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.40f, 0.40f, 0.80f, 0.20f);

        static bool no_titlebar = false;
        static bool no_resize = false;
        static bool no_move = false;
        static bool no_scrollbar = false;
        static bool no_collapse = false;
        static bool no_menu = true;

        // Demonstrate the various window flags. Typically you would just use the default.
        ImGuiWindowFlags window_flags = 0;
        if (no_titlebar)  window_flags |= ImGuiWindowFlags_NoTitleBar;
        if (no_resize)    window_flags |= ImGuiWindowFlags_NoResize;
        if (no_move)      window_flags |= ImGuiWindowFlags_NoMove;
        if (no_scrollbar) window_flags |= ImGuiWindowFlags_NoScrollbar;
        if (no_collapse)  window_flags |= ImGuiWindowFlags_NoCollapse;
        if (!no_menu)     window_flags |= ImGuiWindowFlags_MenuBar;

        // Always center this window when appearing
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));

        char str[128];
        sprintf(str, "Control##%s", win_id.c_str());
        ImGui::SetNextWindowSize(ImVec2(640 * vis_xscale, 520 * vis_xscale), ImGuiCond_FirstUseEver);
        bool is_window_opened = true;
        ImGui::Begin(str, &is_window_opened, window_flags);
        {
            // print selected camera info.
            show_panel_print_selected_camera_info();

            ImGui::SameLine();

            // print FPS.
            show_panel_print_fps();

            ImGui::Separator();

            // movie rec.
            show_panel_movie_rec();

            ImGui::Separator();

            show_panel_shutter_control();
            show_panel_white_balance_control();
            show_panel_iso_sensitivity_control();
            show_panel_f_number_control();
            // ND.
            // FPS.

            // PTZ
            // focus.
            // Stream setting.

            if (ImGui::Button("Live View")) {
                is_display_image = true;                
            }
        }
        ImGui::End();

        return is_window_opened;
    }

    bool display_live_view(const std::string &win_id)
    {
        bool ret = false;
        if (is_display_image) ret = show_panel_live_view();

        return ret;
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

        static bool no_titlebar = false;
        static bool no_resize = false;
        static bool no_move = false;
        static bool no_scrollbar = false;
        static bool no_collapse = false;
        static bool no_menu = true;

        // Demonstrate the various window flags. Typically you would just use the default.
        ImGuiWindowFlags window_flags = 0;
        if (no_titlebar)  window_flags |= ImGuiWindowFlags_NoTitleBar;
        if (no_resize)    window_flags |= ImGuiWindowFlags_NoResize;
        if (no_move)      window_flags |= ImGuiWindowFlags_NoMove;
        if (no_scrollbar) window_flags |= ImGuiWindowFlags_NoScrollbar;
        if (no_collapse)  window_flags |= ImGuiWindowFlags_NoCollapse;
        if (!no_menu)     window_flags |= ImGuiWindowFlags_MenuBar;

        // Always center this window when appearing
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));

        char str[128];
        sprintf(str, "SSH Login##%s", win_id.c_str());
        ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
        bool is_window_opened = true;
        ImGui::Begin(str, &is_window_opened, window_flags);
        {
            // print selected camera info.
            show_panel_print_selected_camera_info();

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

            // if CGI is OK?
            {
                auto is_update = cgi->is_update_cmd_info();

                if (is_update) cgi->fetch();

                is_window_opened = display_camera_window(win_id);

                if (is_update) cgi->next();
            }

            is_display_image = display_live_view(win_id);
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

        return ret && ret_inq && ret_set;
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
