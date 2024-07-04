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

// Main code
int main(int ac, char *av[])
{
    auto win_w = 800;
    auto win_h = 480;

    // initialize GLFW.
    auto glsl_version = init_glfw();

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

    // Go!!
    gui_win->Go();

    // Cleanup
    cleanup_imgui_glfw(window);

    return EXIT_SUCCESS;
}
