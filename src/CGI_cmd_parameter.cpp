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

#include "CGI_cmd_parameter.h"

namespace CGICmd
{

shutter_list exposure_exposure_time_5994p = {
    { 5, "64F" },
    { 6, "32F" },
    { 7, "16F" },
    { 8, "8F" },
    { 9, "7F" },
    { 10, "6F" },
    { 11, "5F" },
    { 12, "4F" },
    { 13, "3F" },
    { 14, "2F" },
    { 15, "1/50" },
    { 16, "1/60" },
    { 17, "1/100" },
    { 18, "1/120" },
    { 19, "1/125" },
    { 20, "1/250" },
    { 21, "1/500" },
    { 22, "1/1000" },
    { 23, "1/2000" },
    { 24, "1/4000" },
    { 25, "1/8000" },
};

shutter_list exposure_exposure_time_5000p = exposure_exposure_time_5994p;

shutter_list exposure_exposure_time_2997p = {
    { 3, "64F" },
    { 4, "32F" },
    { 5, "16F" },
    { 6, "8F" },
    { 7, "7F" },
    { 8, "6F" },
    { 9, "5F" },
    { 10, "4F" },
    { 11, "3F" },
    { 12, "2F" },
    { 13, "1/30" },
    { 14, "1/40" },
    { 15, "1/50" },
    { 16, "1/60" },
    { 17, "1/100" },
    { 18, "1/120" },
    { 19, "1/125" },
    { 20, "1/250" },
    { 21, "1/500" },
    { 22, "1/1000" },
    { 23, "1/2000" },
    { 24, "1/4000" },
    { 25, "1/8000" },
};

shutter_list exposure_exposure_time_2500p = {
    { 3, "64F" },
    { 4, "32F" },
    { 5, "16F" },
    { 6, "8F" },
    { 7, "7F" },
    { 8, "6F" },
    { 9, "5F" },
    { 10, "4F" },
    { 11, "3F" },
    { 12, "2F" },
    { 13, "1/25" },
    { 14, "1/33" },
    { 15, "1/50" },
    { 16, "1/60" },
    { 17, "1/100" },
    { 18, "1/120" },
    { 19, "1/125" },
    { 20, "1/250" },
    { 21, "1/500" },
    { 22, "1/1000" },
    { 23, "1/2000" },
    { 24, "1/4000" },
    { 25, "1/8000" },
};

shutter_list exposure_exposure_time_2400p = {
    { 1, "64F" },
    { 2, "32F" },
    { 3, "16F" },
    { 4, "8F" },
    { 5, "7F" },
    { 6, "6F" },
    { 7, "5F" },
    { 8, "4F" },
    { 9, "3F" },
    { 10, "2F" },
    { 11, "1/24" },
    { 12, "1/32" },
    { 13, "1/48" },
    { 14, "1/50" },
    { 15, "1/60" },
    { 16, "1/96" },
    { 17, "1/100" },
    { 18, "1/120" },
    { 19, "1/125" },
    { 20, "1/250" },
    { 21, "1/500" },
    { 22, "1/1000" },
    { 23, "1/2000" },
    { 24, "1/4000" },
    { 25, "1/8000" },
};

shutter_list exposure_exposure_time_2398p = exposure_exposure_time_2400p;

std::unordered_map<std::string, shutter_list> exposure_exposure_time = {
    { "5994", exposure_exposure_time_5994p },
    { "5000", exposure_exposure_time_5000p },
    { "2997", exposure_exposure_time_2997p },
    { "2500", exposure_exposure_time_2500p },
    { "2400", exposure_exposure_time_2400p },
    { "2398", exposure_exposure_time_2398p },
};

angle_list exposure_angle = {
    { 1, "64F" },
    { 2, "32F" },
    { 3, "16F" },
    { 4, "8F" },
    { 5, "7F" },
    { 6, "6F" },
    { 7, "5F" },
    { 8, "4F" },
    { 9, "3F" },
    { 10, "2F" },
    { 11, "360.0(deg)" },
    { 12, "300.0(deg)" },
    { 13, "270.0(deg)" },
    { 14, "240.0(deg)" },
    { 15, "216.0(deg)" },
    { 16, "210.0(deg)" },
    { 17, "180.0(deg)" },
    { 18, "172.8(deg)" },
    { 19, "150.0(deg)" },
    { 20, "144.0(deg)" },
    { 21, "120.0(deg)" },
    { 22, "90.0(deg)" },
    { 23, "86.4(deg)" },
    { 24, "72.0(deg)" },
    { 25, "45.0(deg)" },
    { 26, "30.0(deg)" },
    { 27, "22.5(deg)" },
    { 28, "11.25(deg)" },
    { 29, "5.6(deg)" },
};

}   // namespace CGICmd
