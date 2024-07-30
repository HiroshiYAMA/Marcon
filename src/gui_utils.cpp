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

#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <thread>

#include "gui_utils.h"

// visualized scale of display.
float vis_xscale, vis_yscale;

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

std::string init_glfw()
{
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        std::exit(EXIT_FAILURE);

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    return std::string(glsl_version);
}

void setup_imgui(GLFWwindow* window, const char *glsl_version)
{
    glfwGetWindowContentScale(window, &vis_xscale, &vis_yscale);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();
    // ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != NULL);

    ImGui::GetStyle().ScaleAllSizes(vis_xscale);
    ImFontConfig font_cfg;
    font_cfg.SizePixels = 13.0f * vis_xscale;
    io.Fonts->AddFontDefault(&font_cfg);
    io.FontGlobalScale = app_font_scale;
}

void cleanup_imgui_glfw(GLFWwindow* window)
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

// フルスクリーンモードにする(メイン映像の見た目はウィンドウモード).
void setDisplayModeFullscreen(GLFWwindow* window)
{
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *vmode = glfwGetVideoMode(monitor);
    glfwSetWindowMonitor(window, monitor, 0, 0, vmode->width, vmode->height, vmode->refreshRate);	// フルスクリーン //

    // ImGui::StyleColorsDark();
    ImGui::GetStyle().FrameBorderSize = 1.0f;
}

// ウィンドウモードにする(メイン映像の見た目はフルスクリーンモード).
void setDisplayModeWindow(GLFWwindow* window, GLsizei width, GLsizei height)
{
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *vmode = glfwGetVideoMode(monitor);
    auto aspect = GLfloat(width) / GLfloat(height);
    adjust_window_size(width, height, vmode->width, vmode->height, aspect, 0.9f);
    GLsizei pos_w = (vmode->width - width) / 2;
    GLsizei pos_h = (vmode->height - height) / 2;
    glfwSetWindowMonitor(window, NULL, pos_w, pos_h, width, height, vmode->refreshRate);
    glfwSetWindowAspectRatio(window, width, height);	// アスペクト比固定 //

    // ImGui::StyleColorsClassic();
    ImGui::GetStyle().FrameBorderSize = 1.0f;
}

// OpenGL のテクスチャの型を得る.
GLenum get_tex_type(int pixel_depth, bool pixel_float)
{
    GLenum tex_type = (pixel_float
        ? (pixel_depth > 10 ? GL_FLOAT : GL_HALF_FLOAT)
        : (pixel_depth > 8 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE));

    return tex_type;
}

std::tuple<GLint, GLenum> get_format(const GLenum type, const int num)
{
    GLint internalFormat;
    GLenum format;

    if (type == GL_FLOAT) {
        internalFormat = (num == 1 ? GL_R32F : (num == 2 ? GL_RG32F : (num == 3 ? GL_RGB32F : GL_RGBA32F)));

    } else if (type == GL_HALF_FLOAT) {
        internalFormat = (num == 1 ? GL_R16F : (num == 2 ? GL_RG16F : (num == 3 ? GL_RGB16F : GL_RGBA16F)));

    } else if (type == GL_UNSIGNED_SHORT) {
        internalFormat = (num == 1 ? GL_R16 : (num == 2 ? GL_RG16 : (num == 3 ? GL_RGB16 : GL_RGBA16)));

    } else {
        internalFormat = (num == 1 ? GL_R8 : (num == 2 ? GL_RG8 : (num == 3 ? GL_RGB8 : GL_RGBA8)));
    }

    format = (num == 1 ? GL_RED : (num == 2 ? GL_RG : (num == 3 ? GL_RGB : GL_RGBA)));

    return std::make_tuple(internalFormat, format);
};

// target: GL_TEXTURE_2D, GL_TEXTURE_3D.
// type: GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_HALF_FLOAT, GL_FLOAT.
// num: 1, 2, 3, 4.
// filter: GL_NEAREST, GL_LINEAR.
// wrap: GL_CLAMP_TO_EDGE, GL_CLAMP_TO_BORDER, GL_MIRRORED_REPEAT, GL_REPEAT, GL_MIRROR_CLAMP_TO_EDGE.
GLuint make_texture(GLuint tex_id, GLenum target, GLsizei width, GLsizei height, GLsizei depth, GLenum type, int num, GLint filter, GLint wrap)
{
    auto get_format = [](const GLenum type, const int num) -> std::tuple<GLint, GLenum> {
        GLint internalFormat;
        GLenum format;

        if (type == GL_FLOAT) {
            internalFormat = (num == 1 ? GL_R32F : (num == 2 ? GL_RG32F : (num == 3 ? GL_RGB32F : GL_RGBA32F)));

        } else if (type == GL_HALF_FLOAT) {
            internalFormat = (num == 1 ? GL_R16F : (num == 2 ? GL_RG16F : (num == 3 ? GL_RGB16F : GL_RGBA16F)));

        } else if (type == GL_UNSIGNED_SHORT) {
            internalFormat = (num == 1 ? GL_R16 : (num == 2 ? GL_RG16 : (num == 3 ? GL_RGB16 : GL_RGBA16)));

        } else {
            internalFormat = (num == 1 ? GL_R8 : (num == 2 ? GL_RG8 : (num == 3 ? GL_RGB8 : GL_RGBA8)));
        }

        format = (num == 1 ? GL_RED : (num == 2 ? GL_RG : (num == 3 ? GL_RGB : GL_RGBA)));

        return std::make_tuple(internalFormat, format);
    };

    auto [internalFormat, format] = get_format(type, num);

    if (tex_id > 0) glDeleteTextures(1, &tex_id);
    glGenTextures(1, &tex_id);
    glBindTexture(target, tex_id);
    if (target == GL_TEXTURE_3D) {
        // width, height, depth: max size is 2048. //
        glTexImage3D(target, 0, internalFormat, width, height, depth, 0, format, type, NULL);
    }
    else {	// GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE を想定 //
        // width, height: max size is 8192. //
        glTexImage2D(target, 0, internalFormat, width, height, 0, format, type, NULL);
    }
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_borderColor);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap);
    if (target == GL_TEXTURE_3D) {
        glTexParameteri(target, GL_TEXTURE_WRAP_R, wrap);
    }

    //if (glewGetExtension("GL_EXT_pixel_transform")) {
    //	std::cout << "[Enable]: GL_EXT_pixel_transform" << std::endl;
    //	glPixelTransformParameteriEXT(GL_PIXEL_TRANSFORM_2D_EXT, GL_PIXEL_MAG_FILTER_EXT, GL_CUBIC_EXT);
    //	glPixelTransformParameteriEXT(GL_PIXEL_TRANSFORM_2D_EXT, GL_PIXEL_MIN_FILTER_EXT, GL_AVERAGE_EXT);
    //} else {
    //	std::cout << "[Disable]: GL_EXT_pixel_transform" << std::endl;
    //}

    //// 異方性フィルタリング.
    //if (glewGetExtension("GL_EXT_texture_filter_anisotropic")) {
    //	std::cout << "[Enable]: GL_EXT_texture_filter_anisotropic" << std::endl;
    //	int max_anisotropy;
    //	glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_anisotropy);
    //	glTexParameteri(target, GL_TEXTURE_MAX_ANISOTROPY_EXT, max_anisotropy);
    //} else {
    //	std::cout << "[Disable]: GL_EXT_texture_filter_anisotropic" << std::endl;
    //}

    return tex_id;
}

bool judge_aspect_wh(float window_aspect, float monitor_aspect)
{
    // true: 横(width)がはみ出る、 false: 縦(height)がはみ出る.
    return (monitor_aspect <= window_aspect);
}

void adjust_window_size(GLsizei &win_w, GLsizei &win_h, GLsizei mon_w, GLsizei mon_h, GLfloat aspect, GLfloat margin)
{
    float aspect_mon = float(mon_w) / float(mon_h);
    float aspect_win = float(win_w) / float(win_h);
    auto adj_margin = [&](GLsizei s) -> GLsizei { return GLsizei(s * margin); };

    if (judge_aspect_wh(aspect_win, aspect_mon)) {	// 横がはみ出るかも //
        if (adj_margin(mon_w) < win_w) {
            win_w = adj_margin(mon_w);
            win_h = GLsizei(win_w / aspect);
        }
    } else {	// 縦がはみ出るかも //
        if (adj_margin(mon_h) < win_h) {
            win_h = adj_margin(mon_h);
            win_w = GLsizei(win_h * aspect);
        }
    }
    win_w = std::max(win_w, 1);
    win_h = std::max(win_h, 1);
}

// テクスチャ(N枚)を表示する //
void display_textures(GLuint num, const GLuint *texIDs, GLsizei width, GLsizei height, std::string title,
    bool *p_open, ImGuiWindowFlags window_flags,
    bool mouse_through, bool orientation_flag, bool v_flip_flag,
    const std::vector<std::string> info_str)
{
    window_flags |= ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_NoSavedSettings;

    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameRounding = 8.0f * vis_xscale;
    style.GrabRounding = 16.0f * vis_xscale;

    const ImVec2 canvas_margin(18, 36);
    const auto aspect = GLfloat(width) / GLfloat(height);
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *vmode = glfwGetVideoMode(monitor);
    auto width_max = vmode->width - GLsizei(canvas_margin.x);
    auto height_max = vmode->height - GLsizei(canvas_margin.y);
    adjust_window_size(width, height, width_max, height_max, aspect, 1.0f);
    ImVec2 canvas_size = ImVec2(width, height);

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(canvas_size.x + canvas_margin.x, canvas_size.y + canvas_margin.y), ImGuiCond_Once);
    if (ImGui::Begin(title.c_str(), p_open, window_flags)) {
        // ImDrawList* draw_list = ImGui::GetWindowDrawList();

        const ImVec2 p = ImGui::GetCursorScreenPos();
        const ImVec2 win_size = ImGui::GetWindowSize();
        width = win_size.x - canvas_margin.x;
        height = win_size.y - canvas_margin.y;
        auto aspect_win = GLfloat(width) / GLfloat(height);
        if ((aspect > 1.0f && aspect_win < 1.0f) || (aspect < 1.0f && aspect_win > 1.0f)) {
            std::swap(width, height);
        }
        if (orientation_flag) {
            width = height * aspect;
        } else {
            height = width / aspect;
        }
        adjust_window_size(width, height, width_max, height_max, aspect, 1.0f);
        canvas_size.x = width;
        canvas_size.y = height;
        ImGui::SetWindowSize(title.c_str(), ImVec2(canvas_size.x + canvas_margin.x, canvas_size.y + canvas_margin.y));

        // // テクスチャと同じサイズのダミーボタンを置いて、パンチルトなどのマウス操作が効くようにする.
        // if (mouse_through) {
        //     std::string IDstr = "canvas##" + title;
        //     ImGui::InvisibleButton(IDstr.c_str(), canvas_size);
        //     flag_movie_canvas_hovered = ImGui::IsItemHovered();	// マウスカーソルのホバー判定 //
        // }

        // テクスチャを表示.
        auto dh = GLfloat(canvas_size.y) / GLfloat(num);
        const auto uv_a = v_flip_flag ? ImVec2(0, 0) : ImVec2(0, 1);
        const auto uv_b = v_flip_flag ? ImVec2(1, 1) : ImVec2(1, 0);
        for (auto i = 0; i < num; i++) {
            // draw_list->AddImage((ImTextureID)texIDs[i], ImVec2(p.x, p.y + dh * i), ImVec2(p.x + canvas_size.x, p.y + dh * (i + 1)), uv_a, uv_b);
            ImGui::Image(reinterpret_cast<ImTextureID>(texIDs[i]), ImVec2(width, height), uv_a, uv_b);
        }

        // display information.
        {
            ImGuiStyle& style = ImGui::GetStyle();
            auto fr = style.FrameRounding;
            auto gr = style.GrabRounding;

            style.FrameRounding = 0.0f;
            style.GrabRounding = 0.0f;
            ImGui::SetCursorScreenPos(p);
            for (auto &e : info_str) {
                ImGui::Button(e.c_str());
            }

            style.FrameRounding = fr;
            style.GrabRounding = gr;
        }

        ImGui::End();
    }
}

// テクスチャ(1枚)を表示する //
void display_texture(GLuint texID, GLsizei width, GLsizei height, std::string title,
    bool *p_open, ImGuiWindowFlags window_flags,
    bool mouse_through, bool orientation_flag, bool v_flip_flag,
    const std::vector<std::string> info_str)
{
    display_textures(1, &texID, width, height, title, p_open, window_flags, mouse_through, orientation_flag, v_flip_flag, info_str);
}

// ボタンの色(In-Active, Hovered, Active)を設定する //
void set_style_color(float hue, float sat, float val, float a)
{
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(hue, sat, val, a));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(hue, sat, val, a));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(hue, sat, val, a));
}
void set_style_color(float hue, float a)
{
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(hue, 0.5f, 0.5f, a));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(hue, 0.7f, 0.7f, a));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(hue, 0.9f, 0.9f, a));
}
void set_style_color_inactive(float hue, float a)
{
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(hue, 0.1f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(hue, 0.1f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(hue, 0.1f, 0.5f));
}

// ボタンの色設定を戻す(In-Active, Hovered, Active) //
void reset_style_color() { ImGui::PopStyleColor(3); }

// ON/off ボタンの GUI を表示する //
void put_OnOff_button(const char* button_str, int &flag)
{
    std::string flag_str;
    if (ImGui::Button(button_str)) flag = (flag + 1) % 2;
    if (flag == 1) flag_str = "ON";
    else flag_str = "off";
    ImGui::SameLine();ImGui::Text("%s", flag_str.c_str());
}
void put_OnOff_button(const char* button_str, bool &flag)
{
    std::string flag_str;
    if (ImGui::Button(button_str)) flag = !flag;
    if (flag) flag_str = "ON";
    else flag_str = "off";
    ImGui::SameLine();ImGui::Text("%s", flag_str.c_str());
}

void show_panel_texture_simple(const GLuint tex_id, GLsizei width, GLsizei height, std::string title,
    ImGuiWindowFlags window_flags,
    bool v_flip_flag,
    const std::vector<std::string> info_str)
{
    window_flags |= ImGuiWindowFlags_NoCollapse;

    // テクスチャを表示.
    const auto uv_a = v_flip_flag ? ImVec2(0, 0) : ImVec2(0, 1);
    const auto uv_b = v_flip_flag ? ImVec2(1, 1) : ImVec2(1, 0);
    ImGui::Image(reinterpret_cast<ImTextureID>(tex_id), ImVec2(width, height), uv_a, uv_b);

    // display information.
    {
        ImGuiStyle& style = ImGui::GetStyle();
        auto fr = style.FrameRounding;
        auto gr = style.GrabRounding;

        style.FrameRounding = 0.0f;
        style.GrabRounding = 0.0f;
        for (auto &e : info_str) {
            ImGui::Button(e.c_str());
        }

        style.FrameRounding = fr;
        style.GrabRounding = gr;
    }
}

void show_panel_texture(const GLuint tex_id, GLsizei width, GLsizei height, std::string title,
    ImGuiWindowFlags window_flags,
    bool v_flip_flag,
    const std::vector<std::string> info_str)
{
    window_flags |= ImGuiWindowFlags_NoCollapse;

    const ImVec2 canvas_margin(18, 36);
    const auto aspect = GLfloat(width) / GLfloat(height);
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *vmode = glfwGetVideoMode(monitor);
    auto width_max = vmode->width - GLsizei(canvas_margin.x);
    auto height_max = vmode->height - GLsizei(canvas_margin.y);
    adjust_window_size(width, height, width_max, height_max, aspect, 1.0f);
    ImVec2 canvas_size = ImVec2(width, height);

    if (ImGui::BeginChild(title.c_str(), ImVec2(0, 0), false, window_flags)) {
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const ImVec2 win_size = ImGui::GetWindowSize();
        width = win_size.x - canvas_margin.x;
        height = win_size.y - canvas_margin.y;
        auto aspect_win = GLfloat(width) / GLfloat(height);
        if ((aspect > 1.0f && aspect_win < 1.0f) || (aspect < 1.0f && aspect_win > 1.0f)) {
            std::swap(width, height);
        }
        adjust_window_size(width, height, width_max, height_max, aspect, 1.0f);

        // テクスチャを表示.
        const auto uv_a = v_flip_flag ? ImVec2(0, 0) : ImVec2(0, 1);
        const auto uv_b = v_flip_flag ? ImVec2(1, 1) : ImVec2(1, 0);
        ImGui::Image(reinterpret_cast<ImTextureID>(tex_id), ImVec2(width, width / aspect), uv_a, uv_b);

        // display information.
        {
            ImGuiStyle& style = ImGui::GetStyle();
            auto fr = style.FrameRounding;
            auto gr = style.GrabRounding;

            style.FrameRounding = 0.0f;
            style.GrabRounding = 0.0f;
            ImGui::SetCursorScreenPos(p);
            for (auto &e : info_str) {
                ImGui::Button(e.c_str());
            }

            style.FrameRounding = fr;
            style.GrabRounding = gr;
        }

    }
    ImGui::EndChild();
}

void show_panel_inputtext(const char *id_str, std::string &str, int width, bool is_password, const std::string &str_hint)
{
    ImGui::PushID(id_str);

    ImGui::PushItemWidth(width * vis_xscale * app_font_scale);
    std::string tmp_str = str;
    tmp_str.reserve(256);
    ImGui::InputTextWithHint("##input text", str_hint.c_str(), tmp_str.data(), tmp_str.capacity(), is_password ? ImGuiInputTextFlags_Password : ImGuiInputTextFlags_None);
    ImGui::PopItemWidth();
    str = std::string{ tmp_str.c_str() };

    ImGui::PopID();
}

void show_panel_input_decimal(const char *id_str, std::string &str, int width, const std::string &str_hint)
{
    ImGui::PushID(id_str);

    ImGui::PushItemWidth(width * vis_xscale * app_font_scale);
    std::string tmp_str = str;
    tmp_str.reserve(256);
    ImGui::InputTextWithHint("##input decimal", str_hint.c_str(), tmp_str.data(), tmp_str.capacity(), ImGuiInputTextFlags_CharsDecimal);
    ImGui::PopItemWidth();
    str = std::string{ tmp_str.c_str() };

    ImGui::PopID();
}

void show_panel_input_ip_address(const char *id_str, std::string &str, int width, const std::string &str_hint)
{
    show_panel_input_decimal(id_str, str, width, str_hint);
}

void show_panel_input_port_number(const char *id_str, std::string &str, int width, const std::string &str_hint)
{
    show_panel_input_decimal(id_str, str, width, str_hint);
}

// V 周期待ち.
void waitVperiod(double &t_pre)
{
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *vmode = glfwGetVideoMode(monitor);

    const double v_period = vmode->refreshRate;
    const double dt_1V = 1.0 / v_period;
    double t_cur = glfwGetTime();
    double dt = t_cur - t_pre;

    constexpr double th = 1.0;
    const double t_sleep = std::max(dt_1V - dt, 0.0) * 1000.0 - th;
    const int64_t t = t_sleep;
    if (t > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(t));
    }

    do {
        t_cur = glfwGetTime();
        dt = t_cur - t_pre;
    } while (dt < dt_1V);

    t_pre = t_cur;
}
