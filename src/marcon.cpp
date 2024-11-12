/* MIT License
 *
 *  Copyright (c) 2024 Backcasters.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

#include "Gui_Window.h"
#include "opencv_utils.h"
#include "IpNetwork.h"

constexpr auto APP_WIN_BASE_W = 800;
constexpr auto APP_WIN_BASE_H = 480;

// Main code
int main(int ac, char *av[])
{
    init_search_ipadr();

    // initialize GLFW.
    auto glsl_version = init_glfw();

    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *vmode = glfwGetVideoMode(monitor);
    auto display_w = vmode->width;
    auto display_h = vmode->height;

    auto win_w = display_w;
    auto win_h = display_h;

    float scale_w = float(display_w) / APP_WIN_BASE_W;
    float scale_h = float(display_h) / APP_WIN_BASE_H;

    set_app_font_scale(scale_w);

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(win_w, win_h, "Marcon (w/ Dear ImGui)", nullptr, nullptr);
    if (window == nullptr)
        return EXIT_FAILURE;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // setup imgui.
    setup_imgui(window, glsl_version.c_str());
    glfwSetWindowSize(window, win_w * vis_xscale, win_h * vis_xscale);
    auto gui_win = std::make_unique<Gui_Window>(window, win_w, win_h);

    if (gui_win) {
        auto fn_rs_list = gen_filename_remote_server_list();
        gui_win->load_file_remote_server_list(fn_rs_list);

        // Go!!
        gui_win->Go();

        gui_win->save_file_remote_server_list(fn_rs_list);
    }

    // Cleanup
    cleanup_imgui_glfw(window);

    return EXIT_SUCCESS;
}
