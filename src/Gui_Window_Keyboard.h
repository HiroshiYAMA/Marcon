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

#if __has_include(<charconv>)
#include <charconv>
#endif

class Gui_Window_Keyboard
{
public:
    enum class em_KeyboardPattern {
        LOWER,
        UPPER,
        NUMBER,
        SYMBOL,

        MAX
    };

private:
    static constexpr auto BUTTON_SCALE = 2.0f;
    static constexpr auto BUTTON_SPACE = 12.0f;
    static constexpr auto TEXT_LEN_MAX = 256;

    static constexpr auto KEY_Backspace = "BS";
    static constexpr auto KEY_Enter = "Ent";
    static constexpr auto KEY_LowerUpper = "a/A";
    static constexpr auto KEY_NumberSymbol = "0/@";
    static constexpr auto KEYBOARD_ROW = 3;
    static constexpr auto KEYBOARD_COLUMN = 10;
    em_KeyboardPattern kb_pat = em_KeyboardPattern::LOWER;
    const char *key_pattern[static_cast<int>(em_KeyboardPattern::MAX)][KEYBOARD_ROW][KEYBOARD_COLUMN] = {
        // lower.
        {
            { "a", "b", "c", "d", "e", "f", "g", "h", "i", KEY_Backspace, },
            { "j", "k", "l", "m", "n", "o", "p", "q", "r", KEY_Enter, },
            { "s", "t", "u", "v", "w", "x", "y", "z", " ", KEY_LowerUpper, },
        },

        // upper.
        {
            { "A", "B", "C", "D", "E", "F", "G", "H", "I", KEY_Backspace, },
            { "J", "K", "L", "M", "N", "O", "P", "Q", "R", KEY_Enter, },
            { "S", "T", "U", "V", "W", "X", "Y", "Z", " ", KEY_LowerUpper, },
        },

        // number.
        {
            { "7", "8", "9", ",", "-", "_", " ", " ", " ", KEY_Backspace, },
            { "4", "5", "6", "0", "+", "=", " ", " ", " ", KEY_Enter, },
            { "1", "2", "3", ".", "/", "*", " ", " ", " ", KEY_NumberSymbol, },
        },

        // symbol.
        {
            { "!", "^", "$", "\"", "|", "(", ")", "[", "]", KEY_Backspace, },
            { "~", "%", "#", "'", "\\", "<", ">", "{", "}", KEY_Enter, },
            { "?", "&", "@", "`", "_", ";", ":", " ", " ", KEY_NumberSymbol, },
        },
    };

public:
    bool display_keyboard_window(bool req_open, const std::string &text_name, std::string &text, em_KeyboardPattern kb_pat_init = em_KeyboardPattern::LOWER, bool is_password = false)
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

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImVec2 win_pos(viewport->WorkPos.x * vis_xscale, (viewport->WorkPos.y + viewport->WorkSize.y / 3) * vis_xscale);
        ImVec2 win_size(viewport->WorkSize.x * vis_xscale, viewport->WorkSize.y * vis_xscale * 2 / 3);

        ImGui::SetNextWindowPos(win_pos, ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(win_size, ImGuiCond_Appearing);

        if (req_open) ImGui::OpenPopup("Keyboard");

        bool is_opened = true;
        if (ImGui::BeginPopupModal("Keyboard", &is_opened, window_flags))
        {
            auto is_appering = ImGui::IsWindowAppearing();
            if (is_appering) {
                if (kb_pat_init < em_KeyboardPattern::MAX) kb_pat = kb_pat_init;
            }

            bool is_popup = true;

            ImGuiStyle& style = ImGui::GetStyle();
            const auto pad_frame = style.FramePadding;
            std::string tmp_text = text;
            if (tmp_text.size() > TEXT_LEN_MAX) tmp_text.resize(TEXT_LEN_MAX);
            tmp_text.reserve(TEXT_LEN_MAX);
            auto p = ImGui::GetCursorStartPos();
            ImGui::SetWindowFontScale(1.5f);
            ImGui::Text("%s", text_name.c_str());
            ImGui::SameLine();
            const auto char3_size = ImGui::CalcTextSize("_123_");
            ImGui::PushItemWidth(-char3_size.x);
            ImGui::InputText("##KEYBOARD", tmp_text.data(), tmp_text.capacity(), is_password ? ImGuiInputTextFlags_Password : ImGuiInputTextFlags_None);
            ImGui::PopItemWidth();
            ImGui::SameLine();
            if (ImGui::Button("012", ImVec2(-1, 0))) {
                if (kb_pat == em_KeyboardPattern::LOWER || kb_pat == em_KeyboardPattern::UPPER) kb_pat = em_KeyboardPattern::NUMBER;
                else if (kb_pat == em_KeyboardPattern::NUMBER || kb_pat == em_KeyboardPattern::SYMBOL) kb_pat = em_KeyboardPattern::LOWER;
            }
            ImGui::SetWindowFontScale(1.0f);
            text = std::string{ tmp_text.c_str() };

            const auto text_size = ImGui::CalcTextSize("A");
            auto btn_size_x = (text_size.x + pad_frame.x) * BUTTON_SCALE + BUTTON_SPACE;
            auto btn_size_y = (text_size.y + pad_frame.y) * BUTTON_SCALE + BUTTON_SPACE;
            if (btn_size_x > btn_size_y) btn_size_y = btn_size_x; else btn_size_x = btn_size_y;
            const ImVec2 btn_size(btn_size_x, btn_size_y);
            for (auto j = 0; j < KEYBOARD_ROW; j++) {
                for (auto i = 0; i < KEYBOARD_COLUMN; i++) {
                    auto ch = key_pattern[static_cast<int>(kb_pat)][j][i];
                    ImGui::SetWindowFontScale(strlen(ch) == 1 ? BUTTON_SCALE : 1.414f);
                    if (ImGui::Button(ch, btn_size)) {
                        std::string str(ch);
                        if (str == KEY_LowerUpper) {
                            if (kb_pat == em_KeyboardPattern::LOWER) kb_pat = em_KeyboardPattern::UPPER; else kb_pat = em_KeyboardPattern::LOWER;
                        } else if (str == KEY_NumberSymbol) {
                            if (kb_pat == em_KeyboardPattern::NUMBER) kb_pat = em_KeyboardPattern::SYMBOL; else kb_pat = em_KeyboardPattern::NUMBER;
                        } else if (str == KEY_Backspace) {
                            text.pop_back();
                        } else if (str == KEY_Enter) {
                            is_popup = false;
                        } else {
                            text.append(str);
                        }
                    }
                    ImGui::SameLine();
                    ImGui::SetWindowFontScale(1.0f);
                }
                ImGui::Text("%s", "");
            }

            if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                is_popup = false;
            }

            auto [is_drag_left, mouse_delta] = is_mouse_drag_to_left(ImGuiMouseButton_Left);
            if (is_drag_left) {
                is_popup = false;
            }

            if (!is_popup) {
                is_opened = false;
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        return is_opened;
    }

    Gui_Window_Keyboard() {}

    virtual ~Gui_Window_Keyboard() {}
};
