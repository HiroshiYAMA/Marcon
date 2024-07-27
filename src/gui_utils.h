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
#include <vector>
#include <cmath>

#if defined(_MSC_VER)
#include "glew.h"
#include <GLFW/glfw3.h>
#include "glext.h"
#elif defined(__APPLE__)
#  include "GL/glew.h"
#  define GLFW_INCLUDE_GLCOREARB
#  include <GLFW/glfw3.h>
#else
#ifdef USE_MULTIMEDIA_API
#  include <GL/glew.h>
#endif
#endif

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

namespace {

// 背景色
constexpr GLfloat background[] = { 0.0f, 0.0f, 0.0f, 0.0f };
constexpr GLfloat tex_borderColor[] = { 0.0f, 0.0f, 0.0f, 0.0f };

}

// visualized scale of display.
extern float vis_xscale, vis_yscale;

constexpr auto app_font_scale = 2.0f;

extern std::string init_glfw();
extern void setup_imgui(GLFWwindow* window, const char *glsl_version);
extern void cleanup_imgui_glfw(GLFWwindow* window);

// フルスクリーンモードにする(メイン映像の見た目はウィンドウモード).
extern void setDisplayModeFullscreen(GLFWwindow* window);
// ウィンドウモードにする(メイン映像の見た目はフルスクリーンモード).
extern void setDisplayModeWindow(GLFWwindow* window, GLsizei width, GLsizei height);

// OpenGL のテクスチャの型を得る.
extern GLenum get_tex_type(int pixel_depth, bool pixel_float);
extern std::tuple<GLint, GLenum> get_format(const GLenum type, const int num);

// target: GL_TEXTURE_2D, GL_TEXTURE_3D.
// type: GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_HALF_FLOAT, GL_FLOAT.
// num: 1, 2, 3, 4.
// filter: GL_NEAREST, GL_LINEAR.
// wrap: GL_CLAMP_TO_EDGE, GL_CLAMP_TO_BORDER, GL_MIRRORED_REPEAT, GL_REPEAT, GL_MIRROR_CLAMP_TO_EDGE.
extern GLuint make_texture(GLuint tex_id, GLenum target, GLsizei width, GLsizei height, GLsizei depth, GLenum type, int num, GLint filter = GL_LINEAR, GLint wrap = GL_CLAMP_TO_BORDER);

extern bool judge_aspect_wh(float window_aspect, float monitor_aspect);
extern void adjust_window_size(GLsizei &win_w, GLsizei &win_h, GLsizei mon_w, GLsizei mon_h, GLfloat aspect, GLfloat margin = 1.0f);

// テクスチャ(N枚)を表示する //
extern void display_textures(GLuint num, const GLuint *texIDs, GLsizei width, GLsizei height, std::string title,
    bool *p_open = NULL, ImGuiWindowFlags window_flags = 0,
    bool mouse_through = false, bool orientation_flag = false, bool v_flip_flag = false,
    const std::vector<std::string> info_str = {});

// テクスチャ(1枚)を表示する //
extern void display_texture(GLuint texID, GLsizei width, GLsizei height, std::string title,
    bool *p_open = NULL, ImGuiWindowFlags window_flags = 0,
    bool mouse_through = false, bool orientation_flag = false, bool v_flip_flag = false,
    const std::vector<std::string> info_str = {});


inline auto get_mouse_drag_delta = [](ImGuiMouseButton button, ImVec4 col = ImGui::GetStyleColorVec4(ImGuiCol_Button)) -> ImVec2 {
    ImVec2 delta(0, 0);

    if (ImGui::IsMouseDragging(button)) {
        auto &io = ImGui::GetIO();
        ImGui::GetForegroundDrawList()->AddLine(io.MouseClickedPos[button], io.MousePos, ImGui::GetColorU32(col), 4.0f);
    } else if (ImGui::IsMouseReleased(button)) {
        delta = ImGui::GetMouseDragDelta(button);
    }

    return delta;
};

inline auto get_mouse_drag_delta_rainbow = [](ImGuiMouseButton button) -> ImVec2 {
    auto &io = ImGui::GetIO();
    auto vec = ImVec2(io.MouseClickedPos[button].x - io.MousePos.x, io.MouseClickedPos[button].y - io.MousePos.y);
    auto len = std::sqrt(std::pow(vec.x, 2.0f) + std::pow(vec.y, 2.0f));
    auto hue = len / 140.0f;
    hue = hue - std::floor(hue);
    auto col = (ImVec4)ImColor::HSV(hue, 0.6f, 0.6f);

    auto delta = get_mouse_drag_delta(button, col);

    return delta;
};

inline auto is_mouse_drag_to_left = [](ImGuiMouseButton button) -> std::tuple<bool, ImVec2> {
    bool ret = false;

    auto mouse_delta = get_mouse_drag_delta_rainbow(button);
    auto win_size = ImGui::GetWindowSize();
    if (mouse_delta.x < -win_size.x / 2) {
        ret = true;
    }

    return { ret, mouse_delta };
};

inline auto is_mouse_drag_to_right = [](ImGuiMouseButton button) -> std::tuple<bool, ImVec2> {
    bool ret = false;

    auto mouse_delta = get_mouse_drag_delta_rainbow(button);
    auto win_size = ImGui::GetWindowSize();
    if (mouse_delta.x > win_size.x / 2) {
        ret = true;
    }

    return { ret, mouse_delta };
};


// ボタンの色(In-Active, Hovered, Active)を設定する //
extern void set_style_color(float hue, float sat, float val);
extern void set_style_color(float hue);
extern void set_style_color_inactive(float hue);

// ボタンの色設定を戻す(In-Active, Hovered, Active) //
extern void reset_style_color();

// ボタンの GUI を表示する //
template <typename T> void put_button_repeat(const char* button_str, T &var, T val)
{
    ImGui::PushButtonRepeat(true);
    ImGui::GetIO().KeyRepeatRate = (1.0f / 10.0f);
    if (ImGui::Button(button_str)) var = val;
    ImGui::PopButtonRepeat();
}
template <typename T> void put_button_repeat_add(const char* button_str, T &var, T val)
{
    put_button_repeat(button_str, var, var + val);
}
template <typename T> void put_button_repeat_multi(const char* button_str, T &var, T val)
{
    put_button_repeat(button_str, var, var * val);
}

template <typename T> void put_arrowbutton_repeat(const char* button_str, ImGuiDir dir, T &var, T val)
{
    ImGui::PushButtonRepeat(true);
    ImGui::GetIO().KeyRepeatRate = (1.0f / 10.0f);
    if (ImGui::ArrowButton(button_str, dir)) var = val;
    ImGui::PopButtonRepeat();
}
template <typename T> void put_arrowbutton_repeat_add(const char* button_str, ImGuiDir dir, T &var, T val)
{
    put_arrowbutton_repeat(button_str, dir, var, var + val);
}
template <typename T> void put_arrowbutton_repeat_multi(const char* button_str, ImGuiDir dir, T &var, T val)
{
    put_arrowbutton_repeat(button_str, dir, var, var * val);
}

// ON/off ボタンの GUI を表示する //
extern void put_OnOff_button(const char* button_str, int &flag);
extern void put_OnOff_button(const char* button_str, bool &flag);

extern void show_panel_texture_simple(const GLuint tex_id, GLsizei width, GLsizei height, std::string title,
    ImGuiWindowFlags window_flags = 0,
    bool v_flip_flag = false,
    const std::vector<std::string> info_str = {});
extern void show_panel_texture(const GLuint tex_id, GLsizei width, GLsizei height, std::string title,
    ImGuiWindowFlags window_flags = 0,
    bool v_flip_flag = false,
    const std::vector<std::string> info_str = {});

extern void show_panel_inputtext(const char *id_str, std::string &str, int width = 200, bool is_password = false, const std::string &str_hint = "");
extern void show_panel_input_decimal(const char *id_str, std::string &str, int width = 200, const std::string &str_hint = "");
extern void show_panel_input_ip_address(const char *id_str, std::string &str, int width = 200, const std::string &str_hint = "IP address");
extern void show_panel_input_port_number(const char *id_str, std::string &str, int width = 200, const std::string &str_hint = "Port #");

// V 周期待ち.
extern void waitVperiod(double &t_pre);
