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
#include "Gui_Window_Keyboard.h"
#include "common_utils.h"
#include "gui_utils.h"
#include "RemoteServer.h"
#include "IpNetwork.h"

namespace {

constexpr auto filename_remote_server_List = "Marcon_remote_servers.json";

auto gen_filename_remote_server_list = []() -> std::string {
    std::error_code ec;
    auto tmp_dir = fs::temp_directory_path(ec);
    auto home_dir = getenv("HOME");

    fs::path script_dir = home_dir ? fs::path{home_dir} : tmp_dir;
    auto json_path = script_dir.append(filename_remote_server_List);

    return json_path.string();
};

}

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

    public:
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(
            st_RemoteServerInfo,

            remote_server
        )
    };

private:
    GLFWwindow* window;
    int win_w;
    int win_h;

    ImVec4 clear_color;

    // cudaStream_t m_cuda_stream = NULL;

    std::vector<st_NetInfo> net_info_list;
    int idx_selected_net_info;

    std::map<std::string, st_RemoteServerInfo, std::less<>> remote_server_info_DB;
    st_RemoteServerInfo remote_server_info;

    bool is_loop;
    float mouse_running_for_QUIT;

    TinyTimer tt;

    em_State state;

    bool show_demo_window;

    Gui_Window_Keyboard kb;
    bool display_input_ip_address;
    bool display_input_port_number;

    // blinking tally.
    std::thread thd_blink_tally;
    std::atomic_bool fin_thd_blink_tally;
    float hue_blink_tally_button = 0.0f;
    using tally_color_t = IpNetwork::VISCA_Tally_Command::em_COLOR;
    void blink_tally(const std::string &name, tally_color_t color = tally_color_t::RED)
    {
        std::unique_ptr<IpNetwork::VISCA_Com> visca_com;
        visca_com = IpNetwork::VISCA_Com::Create(name);

        bool is_on = true;
        for (auto i = 0; i < 20; i++) {
            // Send Tally command.
            auto color = tally_color_t::RED;
            visca_com->send_cmd_tally(color, is_on);

            is_on = !is_on;

            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }

        fin_thd_blink_tally.store(true);
    }

    // Menu.
    void show_menu_control()
    {
        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu("Menu"))
            {
                if (ImGui::BeginMenu("Change skin")) {
                    if (ImGui::MenuItem("Dark", NULL)) { ImGui::StyleColorsDark(); gui_skin = em_GuiSkin::DARK; }
                    if (ImGui::MenuItem("Light", NULL)) { ImGui::StyleColorsLight(); gui_skin = em_GuiSkin::LIGHT; }
                    if (ImGui::MenuItem("Classic", NULL)) { ImGui::StyleColorsClassic(); gui_skin = em_GuiSkin::CLASSIC; }
                    ImGui::EndMenu();
                }
                if (ImGui::MenuItem("Demoru?", NULL)) { show_demo_window = true; }
                if (ImGui::MenuItem("Quit", NULL)) { is_loop = false; }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }
    }

    // search & list IP address.
    void show_panel_search_ip(st_RemoteServer &rs)
    {
        ImGui::SetWindowFontScale(1.5f);
        if (ImGui::Button("Search")) {
            net_info_list = search_ipadr();
        }
        ImGui::SameLine();

        // list camera's IP adderss & Name.
        if (net_info_list.size() > 0)
        {
            auto gen_item_str = [](const st_NetInfo &ni) -> std::string {
                std::string str;
                str = ni.nickname + "(" + ni.ipadr + ")";
                return str;
            };

            ImGuiComboFlags flags = ImGuiComboFlags_None;
            flags |= ImGuiComboFlags_NoPreview;
            flags &= ~(ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_WidthFitPreview);
            std::string str_pre_val = "";
            if (idx_selected_net_info >= 0 && idx_selected_net_info < net_info_list.size()) {
                str_pre_val = gen_item_str(net_info_list[idx_selected_net_info]);
            }
            auto combo_preview_value = str_pre_val.c_str();
            if (ImGui::BeginCombo("##NET_INFO_LIST", combo_preview_value, flags))
            {
                for (int n = 0; n < net_info_list.size(); n++)
                {
                    const bool is_selected = (idx_selected_net_info == n);
                    std::string str = gen_item_str(net_info_list[n]);
                    if (ImGui::Selectable(str.c_str(), is_selected)) {
                        rs.ip_address = net_info_list[n].ipadr;
                        rs.nickname = net_info_list[n].nickname;
                        idx_selected_net_info = n;
                    }
                    if (is_selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }
        ImGui::SetWindowFontScale(1.0f);
    }

    void show_panel_connect_camera(st_RemoteServerInfo &rs_info)
    {
        ImGuiStyle& style = ImGui::GetStyle();
        const auto text_size = ImGui::CalcTextSize("(********) 255.255.255.255:65535/[SRT]Listener");
        const auto pad_frame = style.FramePadding;
        const ImVec2 btn_size(text_size.x + pad_frame.x * 2, (text_size.y + pad_frame.y) * 2);

        auto &rs = rs_info.remote_server;
        std::string str = "(" + rs.nickname + ") " + rs.ip_address + ":" + rs.port + " / [SRT]" + (rs.is_srt_listener ? "Listener" : "Caller");

        set_style_color(2.0f / 7.0f);
        if (ImGui::Button(str.c_str(), btn_size)) {
            auto &gui_win_camera = rs_info.handle;
            if (!gui_win_camera) {
                gui_win_camera = std::make_unique<Gui_Window_Camera>(win_w, win_h, rs_info.remote_server);
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
        reset_style_color();
    }

    // blink tally.
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

    // manage remote server DB.
    void show_panel_manage_remote_server_DB()
    {
        ImGui::PushID("Manage_Remote_Server_DB");

        auto &rs_info = remote_server_info;
        auto &rs = rs_info.remote_server;

        show_panel_search_ip(rs);

        ImGui::SameLine();

        auto req_input_ip_address = show_panel_input_ip_address("IP address", rs.ip_address, 10 * 13.0f, "IPv4 adress"); ImGui::SameLine();
        if (req_input_ip_address) {
            display_input_ip_address = true;
        }
        auto req_input_port_number = show_panel_input_port_number("Port number", rs.port, 4 * 13.0f); ImGui::SameLine();
        if (req_input_port_number) {
            display_input_port_number = true;
        }

        // popup keyboard window.
        if (display_input_ip_address) {
            display_input_ip_address = kb.display_keyboard_window(req_input_ip_address, "IP address", rs.ip_address, Gui_Window_Keyboard::em_KeyboardPattern::NUMBER);
        } else if (display_input_port_number) {
            display_input_port_number = kb.display_keyboard_window(req_input_port_number, "Port number", rs.port, Gui_Window_Keyboard::em_KeyboardPattern::NUMBER);
        }

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
                rs.port = "80";
            }
        }

        ImGui::Separator();

        for (auto &[k, v] : remote_server_info_DB) {
            ImGui::PushID(k.c_str());

            show_panel_connect_camera(v);

            ImGui::SameLine();

            set_style_color(6.0f / 7.0f);
            if (ImGui::Button("Del")) {
                remote_server_info_DB.erase(k);

                reset_style_color();
                ImGui::PopID();
                break;
            }
            reset_style_color();

            ImGui::SameLine();

            // blink tally.
            show_panel_blink_tally(v.remote_server.ip_address);

            ImGui::PopID();
        }

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

    bool display_launcher_window()
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

        bool is_opened = true;
        ImGui::Begin("Launcher", &is_opened, window_flags);
        {
            // Menu.
            show_menu_control();

            // manage remote server DB.
            show_panel_manage_remote_server_DB();

            ImGui::Separator();

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

            // Quit by mouse action.
            {
                constexpr auto quit_cnt = 5;

                if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    auto &io = ImGui::GetIO();
                    auto delta = io.MouseDelta;
                    auto len = std::sqrt(std::pow(delta.x, 2.0f) + std::pow(delta.y, 2.0f));
                    mouse_running_for_QUIT += len;
                } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    auto win_size = ImGui::GetWindowSize();
                    if (mouse_running_for_QUIT > win_size.x * quit_cnt) {
                        is_opened = false;
                    } else {
                        mouse_running_for_QUIT = 0.0f;
                    }
                } else {
                    mouse_running_for_QUIT = 0.0f;
                }
            }
        }
        ImGui::End();

        return is_opened;
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

        net_info_list = {};
        idx_selected_net_info = -1;

        remote_server_info_DB = std::move(decltype(remote_server_info_DB){});
        remote_server_info = {};
        remote_server_info.remote_server.port = "80";
        // remote_server_info.remote_server.srt_port = "4201";

        is_loop = true;
        mouse_running_for_QUIT = 0.0f;

        tt = {};

        state = em_State::LANCHER;

        show_demo_window = false;

        kb = {};
        display_input_ip_address = false;
        display_input_port_number = false;

        fin_thd_blink_tally.store(true);

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
        if (thd_blink_tally.joinable()) {
            thd_blink_tally.join();
        }

        DISCONNECT();
    }

    // Main loop
    void Go()
    {
#if 1
        bool full_screen = true;
        setDisplayModeFullscreen(window);
#else
        bool full_screen = false;
#endif

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
                    break;
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
                is_loop &= display_launcher_window();

            } else if (state == em_State::CAMERA_CONTROL) {
                // display control window.
                for (auto &[k, v] : remote_server_info_DB) {
                    auto &gui_win_camera = v.handle;
                    const auto &id = k;
                    if (gui_win_camera) {
                        auto ret = gui_win_camera->display_control_window(id);
                        gui_win_camera->sync_remote_server(v.remote_server);
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

    // convert remote server database -> json.
    njson rsDB2json()
    {
        njson js(remote_server_info_DB);

        return js;
    }

    // convert json -> remote server database.
    void json2rsDB(const njson &js)
    {
        for (const auto &e : js.items()) {
            const auto &id = e.key();
            const auto &js_rsDB = e.value();
            st_RemoteServerInfo rsDB;
            from_json(js_rsDB, rsDB);
            remote_server_info_DB[id] = std::move(rsDB);
        }
    }

    // load remote server list file. (<- json).
    void load_file_remote_server_list(const std::string &filename)
    {
        auto js = read_json_file(filename);
        json2rsDB(js);
    }

    // save remote server list file. (-> json).
    void save_file_remote_server_list(const std::string &filename)
    {
        njson js = {};
        js = rsDB2json();

        write_json_file(filename, js);
    }
};
