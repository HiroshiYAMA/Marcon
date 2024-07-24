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
#include <fstream>

#include "Gui_Window_Camera.h"
#include "common_utils.h"
#include "gui_utils.h"
#include "RemoteServer.h"

// namespace {

// constexpr auto app_json_file_path = "RemoteGui.json";

// auto gen_setting_json_file = []() -> std::string {
//     std::error_code ec;
//     auto tmp_dir = fs::temp_directory_path(ec);
//     auto home_dir = getenv("HOME");

//     fs::path script_dir = home_dir ? fs::path{home_dir} : tmp_dir;
//     auto json_path = script_dir.append(app_json_file_path);

//     return json_path.string();
// };

// }

class Gui_Window
{
    enum class em_State {
        LANCHER,
        CAMERA_CONTROL,
    };

    struct st_RemoteServerInfo
    {
        std::unique_ptr<Gui_Window_Camera> handle = nullptr;
        st_RemoteServer remote_server = {};
    };

private:
    GLFWwindow* window;
    int win_w;
    int win_h;

    ImVec4 clear_color;

    // cudaStream_t m_cuda_stream = NULL;

    std::map<std::string, st_RemoteServerInfo, std::less<>> remote_server_info_DB;
    st_RemoteServerInfo remote_server_info;

    bool is_loop;

    TinyTimer tt;

    em_State state;

    bool show_demo_window;

    // Menu.
    void show_menu_control()
    {
        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu("Menu"))
            {
                if (ImGui::MenuItem("Demoru?", NULL)) { show_demo_window = true; }
                if (ImGui::MenuItem("Quit", NULL)) { is_loop = false; }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }
    }

    // manage remote server DB.
    void show_panel_manage_remote_server_DB()
    {
        ImGui::PushID("Manage_Remote_Server_DB");

        auto &rs_info = remote_server_info;
        auto &rs = rs_info.remote_server;
        show_panel_input_ip_address("IP address", rs.ip_address, 10 * 13.0f, "IPv4 adress"); ImGui::SameLine();
        show_panel_input_port_number("Port number", rs.port, 4 * 13.0f); ImGui::SameLine();
        if (ImGui::Button("Add")) {
            if (rs.ip_address != "" && rs.port != "") {
                auto &key = rs.ip_address;
#if __cplusplus == 202002L  // C++20.
                auto is_exist = remote_server_info_DB.contains(key);
#else
                auto &vec = remote_server_info_DB;
                auto is_exist = (std::find_if(vec.begin(), vec.end(), [&key](auto &e){ return e.first == key; }) != vec.end());
#endif
                if (!is_exist) {
                    rs_info.handle.reset();
                    remote_server_info_DB.emplace(key, std::move(rs_info));
                }
                rs_info.handle.reset();
                rs = {};
            }
        }

        ImGui::Separator();

        set_style_color(2.0f / 7.0f);

        for (auto &[k, v] : remote_server_info_DB) {
            ImGui::PushID(k.c_str());

            auto &rs = v.remote_server;
            std::string str = rs.ip_address + ":" + rs.port + " / [SRT]" + (rs.is_srt_listener ? "Listener" : "Caller");
            if (ImGui::Button(str.c_str())) {
                auto &gui_win_camera = v.handle;
                if (!gui_win_camera) {
                    gui_win_camera = std::make_unique<Gui_Window_Camera>(win_w, win_h, v.remote_server);
                    if (gui_win_camera) {
                        if (gui_win_camera->is_CONNECTED()) {
                            state = em_State::CAMERA_CONTROL;
                        } else {
                            gui_win_camera->DISCONNECT();
                            gui_win_camera.reset();
                            state = em_State::LANCHER;
                        }
                    }
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Delete")) {
                remote_server_info_DB.erase(k);

                ImGui::PopID();
                break;
            }

            ImGui::PopID();
        }

        reset_style_color();

        // for (auto i = 0; i < remote_server_DB.size(); i++) {
        //     ImGui::PushID(i);

        //     bool &chk = camera_list[i].select;
        //     ImGui::Checkbox("##SELECT", &chk); ImGui::SameLine();

        //     auto &gui_win_camera = camera_list[i].handle;
        //     const auto &name = camera_list[i].name;
        //     const auto &nickname = camera_list[i].nickname;
        //     const auto &username = camera_list[i].username;
        //     const auto &password = camera_list[i].password;
        //     std::string str = (nickname.empty() || nickname == "") ? name : nickname;

        //     if (ImGui::Button(str.c_str())) {
        //         // connect.
        //         if (!gui_win_camera) {
        //             gui_win_camera = std::make_unique<Gui_Window_Camera>(streaming_ndi_width, streaming_ndi_height, is_sdk_server_enable, m_cuda_stream);
        //             gui_win_camera->set_request_connect(i, name, nickname, username, password);

        //         // disconnect.
        //         } else if (gui_win_camera->is_CONNECTED() || gui_win_camera->is_SSH_LOGIN()) {
        //             gui_win_camera->DISCONNECT();
        //             gui_win_camera.reset();
        //         }
        //     } else {
        //         if (gui_win_camera && gui_win_camera->is_CONNECTED()) {
        //             gui_win_camera->update_selected_camera(name);
        //             camera_list[i].nickname = gui_win_camera->get_nickname();
        //             camera_list[i].username = gui_win_camera->get_username();
        //             camera_list[i].password = gui_win_camera->get_password();
        //         }
        //     }
        //     ImGui::SameLine();
        //     auto get_string_camera_connection_status = [&]() -> std::tuple<const char *, ImVec4> {
        //         auto color_disconnected = ImVec4{1, 0, 0, 1};
        //         auto color_ssh_login    = ImVec4{0, 1, 1, 1};
        //         auto color_connecting   = ImVec4{1, 1, 0, 1};
        //         auto color_connected    = ImVec4{0, 1, 0, 1};
        //         auto color_others       = ImVec4{1, 1, 1, 1};

        //         if (!gui_win_camera) return {"Disconnected", color_disconnected};

        //         switch (gui_win_camera->get_camera_connection_stat()) {
        //         case em_Camera_Connection_State::DISCONNECTED:
        //             return {"Disconnected", color_disconnected};
        //         case em_Camera_Connection_State::SSH_LOGIN:
        //             return {"SSH login ...", color_ssh_login};
        //         case em_Camera_Connection_State::CONNECTING:
        //             return {"Connecting ...", color_connecting};
        //         case em_Camera_Connection_State::CONNECTED:
        //             return {"Connected", color_connected};
        //         default:
        //             return {"---", color_others};
        //         }
        //     };
        //     auto [str_stat, str_color] = get_string_camera_connection_status();
        //     ImGui::TextColored(str_color, "%s", str_stat);

        //     ImGui::PopID();
        // }

        ImGui::PopID();
    }

    void display_launcher_window()
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
        window_flags = window_flags | ImGuiWindowFlags_MenuBar;

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImVec2 win_pos(viewport->WorkPos.x * vis_xscale, viewport->WorkPos.y * vis_xscale);
        ImVec2 win_size(viewport->WorkSize.x * vis_xscale, viewport->WorkSize.y * vis_xscale);

        // ImGui::SetNextWindowPos(ImVec2(0 * vis_xscale, 0 * vis_xscale), ImGuiCond_Appearing);
        // ImGui::SetNextWindowSize(ImVec2(800 * vis_xscale, 480 * vis_xscale), ImGuiCond_Appearing);
        ImGui::SetNextWindowPos(win_pos, ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(win_size, ImGuiCond_Appearing);

        ImGui::Begin("Launcher", NULL, window_flags);
        {
            // Menu.
            show_menu_control();

            // manage remote server DB.
            show_panel_manage_remote_server_DB();

            ImGui::Separator();

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        }
        ImGui::End();
    }

	// // ウィンドウのサイズ変更時の処理
	// //   ・ウィンドウのサイズ変更時にコールバック関数として呼び出される
	// //   ・ウィンドウの作成時には明示的に呼び出す
	// static void resize(GLFWwindow *window, int width, int height);

	// // マウスボタンを操作したときの処理
	// //   ・マウスボタンを押したときにコールバック関数として呼び出される
	// static void mouse(GLFWwindow *window, int button, int action, int mods);

	// // マウスホイール操作時の処理
	// //   ・マウスホイールを操作した時にコールバック関数として呼び出される
	// static void wheel(GLFWwindow *window, double x, double y);

	// // キーボードをタイプした時の処理
	// //   ・キーボードをタイプした時にコールバック関数として呼び出される
	// static void keyboard(GLFWwindow *window, int key, int scancode, int action, int mods);

	// // ドラッグ&ドロップした時の処理
	// //   ・ドラッグ&ドロップした時にコールバック関数として呼び出される
	// static void drag_drop(GLFWwindow* window, int count, const char** paths);

protected:

public:
    Gui_Window(GLFWwindow* _window, int _width, int _height)
    {
        window = _window;
        win_w = _width;
        win_h = _height;

        clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

        // // if (CUDA_FAILED(cudaStreamCreateWithFlags(&m_cuda_stream, cudaStreamNonBlocking))) {
        // //     std::cout << "Failed getting CUDA Steram." << std::endl;
        //     m_cuda_stream = NULL;
        // // }

        remote_server_info_DB = std::move(decltype(remote_server_info_DB){});
        remote_server_info = {};
        remote_server_info.remote_server.port = "80";
        // remote_server_info.remote_server.srt_port = "4201";

        is_loop = true;

        tt = {};

        state = em_State::LANCHER;

        show_demo_window = false;

        // このインスタンスの this ポインタを記録しておく
        glfwSetWindowUserPointer(window, this);

        // // ウィンドウのサイズ変更時に呼び出す処理の登録
        // glfwSetFramebufferSizeCallback(window, resize);

        // // マウスボタンを操作したときの処理
        // glfwSetMouseButtonCallback(window, mouse);

        // // マウスホイール操作時に呼び出す処理
        // glfwSetScrollCallback(window, wheel);

        // // キーボードを操作した時の処理
        // glfwSetKeyCallback(window, keyboard);

        // // ドラッグ&ドロップした時の処理
        // glfwSetDropCallback(window, drag_drop);
    }

    virtual ~Gui_Window()
    {
        DISCONNECT();
    }

    // Main loop
    void Go()
    {
        bool full_screen = true;
        setDisplayModeFullscreen(window);

        while (!glfwWindowShouldClose(window) && is_loop)
        {
            // Poll and handle events (inputs, window resize, etc.)
            // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
            // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
            // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
            // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
            glfwPollEvents();

            {
                // Quit program. key: Shift + Q.
                if (ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_ModShift)) {
                    is_loop = false;
                }

                // switch full screen / window. key: Enter.
                static auto k_enter_pre = false;
                auto k_enter = ImGui::IsKeyPressed(ImGuiKey_F8);
                if (!k_enter_pre && k_enter) {
                    full_screen = !full_screen;
                    if (full_screen) {
                        setDisplayModeFullscreen(window);
                    } else {
                        setDisplayModeWindow(window, win_w * vis_xscale, win_h * vis_xscale);
                    }
                }
                k_enter_pre = k_enter;
            }

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
            if (show_demo_window)
                ImGui::ShowDemoWindow(&show_demo_window);

            if (state == em_State::LANCHER) {
                // display launcher window.
                display_launcher_window();

            } else if (state == em_State::CAMERA_CONTROL) {
                // display control window.
                for (auto &[k, v] : remote_server_info_DB) {
                    auto &gui_win_camera = v.handle;
                    const auto &id = k;
                    if (gui_win_camera) {
                        auto ret = gui_win_camera->display_control_window(id);
                        if (!ret) {
                            gui_win_camera->DISCONNECT();
                            gui_win_camera.reset();
                            state = em_State::LANCHER;
                        }
                    }
                }
            }

            // Rendering
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);

            // V 周期待ち.
            {
                GLFWmonitor *monitor = glfwGetPrimaryMonitor();
                const GLFWvidmode *vmode = glfwGetVideoMode(monitor);
                tt.wait1period(1.0 / vmode->refreshRate);
            }
        }
    }

    void DISCONNECT()
    {
        for (auto &[k, v] : remote_server_info_DB) {
            auto &gui_win_camera = v.handle;
            if (gui_win_camera) {
                gui_win_camera->DISCONNECT();
                gui_win_camera.reset();
            }
        }
    }

    // // convert camera database -> json.
    // njson camDB2json()
    // {
    //     njson js(camera_DB);

    //     return js;
    // }

    // // convert json -> camera database.
    // void json2camDB(const njson &js)
    // {
    //     for (const auto &e : js.items()) {
    //         const auto &id = e.key();
    //         const auto &js_camDB = e.value();
    //         st_CameraDataBase camDB;
    //         from_json(js_camDB, camDB);
    //         camera_DB[id] = camDB;
    //     }
    // }

    // // load setting file. (<- json).
    // void load_setting_file(const std::string &filename)
    // {
    //     std::ifstream ifs(filename);
    //     if (!ifs.is_open()) {
    //         std::cout << "ERROR! can't open JSON file to read : (" << filename << ")" << std::endl;
    //     } else {
    //         njson js = {};
    //         ifs >> js;
    //         json2camDB(js);
    //     }
    // }

    // // save setting file. (-> json).
    // void save_setting_file(const std::string &filename)
    // {
    //     njson js = {};
    //     js = camDB2json();

    //     std::ofstream ofs(filename);
    //     if (!ofs.is_open()) {
    //         std::cout << "ERROR! can't open JSON file to write : (" << filename << ")" << std::endl;
    //     } else {
    //         ofs << std::setw(4) << js << std::endl;
    //     }
    // }
};
