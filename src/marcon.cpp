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

#include "common_utils.h"
#include "gui_utils.h"
#include "opencv_utils.h"

// Main code
int main(int, char**)
{
    auto win_w = 800;
    auto win_h = 480;

    // initialize GLFW.
    auto glsl_version = init_glfw();

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // setup imgui.
    setup_imgui(window, glsl_version.c_str());
    glfwSetWindowSize(window, win_w * vis_xscale, win_h * vis_xscale);

    {
        // Our state
        bool show_demo_window = true;
        bool show_another_window = false;
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

        // Main loop
        bool is_loop = true;
        bool full_screen = false;
        TinyTimer tt;
        while (!glfwWindowShouldClose(window) && is_loop)
        {
            // Poll and handle events (inputs, window resize, etc.)
            // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
            // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
            // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
            // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
            glfwPollEvents();

            {
                auto io = ImGui::GetIO();

                // Quit program. key: Shift + Q.
                if (io.KeysDown[ImGuiKey_Q] && io.KeyShift) {
                    is_loop = false;
                }

                // switch full screen / window. key: Enter.
                static auto k_enter_pre = false;
                auto k_enter = io.KeysDown[ImGuiKey_Enter];
                if (!k_enter_pre && k_enter) {
                    full_screen = !full_screen;
                    if (full_screen) {
                        setDisplayModeFullscreen(window);
                    } else {
                        setDisplayModeWindow(window, win_w, win_h);
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

            // Rendering
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);

            {
                GLFWmonitor *monitor = glfwGetPrimaryMonitor();
                const GLFWvidmode *vmode = glfwGetVideoMode(monitor);
                tt.wait1period(1.0 / vmode->refreshRate);
            }
        }
    }

    // Cleanup
    cleanup_imgui_glfw(window);

    return EXIT_SUCCESS;
}
