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


#if defined(USE_EXPERIMENTAL_FS)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#if defined(__APPLE__)
#include <unistd.h>
#endif
#endif

#include "json.hpp"
using njson = nlohmann::json;

namespace {

// // JSON. Specializing enum conversion.
// NLOHMANN_JSON_SERIALIZE_ENUM( em_COLLO_lens_spec, {
// 	{ em_COLLO_lens_spec::NORMAL, "NORMAL" },
// 	{ em_COLLO_lens_spec::FISHEYE_EQUIDISTANT, "FISHEYE_EQUIDISTANT" },
// 	{ em_COLLO_lens_spec::FISHEYE_EQUISOLID_ANGLE, "FISHEYE_EQUISOLID_ANGLE" },
// 	{ em_COLLO_lens_spec::FISHEYE_ORTHOGRAPHIC, "FISHEYE_ORTHOGRAPHIC" },
// 	{ em_COLLO_lens_spec::FISHEYE_STEREOGRAPHIC, "FISHEYE_STEREOGRAPHIC" },
// })
// NLOHMANN_JSON_SERIALIZE_ENUM( OutputViewType, {
// 	{ IMG_BLEND, "IMG_BLEND" },
// 	{ IMG_MASK, "IMG_MASK" },
// 	{ IMG_INPUT, "IMG_INPUT" },
// 	{ IMG_INPUT_PANORAMA, "IMG_INPUT_PANORAMA" },
// 	{ IMG_BGR, "IMG_BGR" },
// 	{ IMG_FISHEYE, "IMG_FISHEYE" },
// 	{ IMG_FLASH, "IMG_FLASH" },
// 	{ IMG_INPUT_ADJ_COLOR, "IMG_INPUT_ADJ_COLOR" },
// })


// get value from json.
inline auto json_get_val = [](const auto &j, const auto &key, auto &val) -> void {
    auto it = j.find(key);
    if (it != j.end()) {
        njson j_tmp = it.value();
        val = j_tmp.template get<std::remove_reference_t<decltype(val)>>();
    }
};

// get arrayed value from json.
inline auto json_get_array_val = [](auto &j, auto &key, auto &ary) -> void {
    njson j_sub;
    json_get_val(j, key, j_sub);
    auto i = 0;
    for (auto &e : j_sub) {
        if (i >= std::size(ary)) break;
        ary[i] = j_sub.at(i).template get<std::remove_reference_t<decltype(ary[0])>>();
        i++;
    }
};

// get vector value from json.
inline auto json_get_vector_val = [](auto &j, auto &key, auto &vec) -> void {
    njson j_sub;
    json_get_val(j, key, j_sub);
    vec.clear();
    for (auto &e : j_sub) vec.push_back(e.template get<std::remove_reference_t<decltype(vec[0])>>());
};

// convert enum -> std::string.
inline auto json_conv_enum2str = [](auto val) -> std::string {
    njson json = val;
    std::string str = json.template get<std::string>();
    return str;
};

template<typename T> T read_json_file(const std::string &filename)
{
    auto p = fs::path{filename};
    auto ext = p.extension();
    auto ext_str = ext.generic_string();
    std::transform(ext_str.cbegin(), ext_str.cend(), ext_str.begin(), ::tolower);

    njson json = {};

    if (ext_str == ".json") {
        std::ifstream ifs(filename);

        if (!ifs.is_open()) {
            std::cout << "ERROR! can't open JSON file to read : (" << filename << ")" << std::endl;
            return {};
        }
        ifs >> json;

    } else if (ext_str == ".dat" || ext_str == ".cbor") {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            std::cout << "ERROR!! can't open DAT file to read : (" << filename << ")" << std::endl;
            return {};
        }
        auto sz = fs::file_size(p);
        std::vector<uint8_t> cbor(sz);
        ifs.read(reinterpret_cast<char *>(cbor.data()), sz);
        json = njson::from_cbor(cbor);

    } else {
        std::cout << "ERROR! not support file type to read: " << ext_str << "." << std::endl;
        return {};
    }

    T data = json.template get<T>();

    return data;
}

njson read_json_file(const std::string &filename)
{
    auto p = fs::path{filename};
    auto ext = p.extension();
    auto ext_str = ext.generic_string();
    std::transform(ext_str.cbegin(), ext_str.cend(), ext_str.begin(), ::tolower);

    njson json = {};

    if (ext_str == ".json") {
        std::ifstream ifs(filename);

        if (!ifs.is_open()) {
            std::cout << "ERROR! can't open JSON file to read : (" << filename << ")" << std::endl;
            return {};
        }
        ifs >> json;

    } else if (ext_str == ".dat" || ext_str == ".cbor") {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            std::cout << "ERROR!! can't open DAT file to read : (" << filename << ")" << std::endl;
            return {};
        }
        auto sz = fs::file_size(p);
        std::vector<uint8_t> cbor(sz);
        ifs.read(reinterpret_cast<char *>(cbor.data()), sz);
        json = njson::from_cbor(cbor);

    } else {
        std::cout << "ERROR! not support file type to read: " << ext_str << "." << std::endl;
        return {};
    }

    return json;
}

template<typename T> void write_json_file(const std::string &filename, const T &data)
{
    auto p = fs::path{filename};
    auto ext = p.extension();
    auto ext_str = ext.generic_string();
    std::transform(ext_str.cbegin(), ext_str.cend(), ext_str.begin(), ::tolower);

    njson json = {};
    json = data;

    if (ext_str == ".json") {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::cout << "ERROR! can't open JSON file to write : (" << filename << ")" << std::endl;
            return;
        }
        ofs << std::setw(4) << json << std::endl;

    } else if (ext_str == ".dat" || ext_str == ".cbor") {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs.is_open()) {
            std::cout << "ERROR!! can't open DAT file to write : (" << filename << ")" << std::endl;
            return;
        }
        auto cbor = njson::to_cbor(json);
        ofs.write(reinterpret_cast<char *>(cbor.data()), cbor.size());

    } else {
        std::cout << "ERROR! not support file type to write : " << ext_str << "." << std::endl;
        return;
    }
}

void write_json_file(const std::string &filename, const njson &json)
{
    auto p = fs::path{filename};
    auto ext = p.extension();
    auto ext_str = ext.generic_string();
    std::transform(ext_str.cbegin(), ext_str.cend(), ext_str.begin(), ::tolower);

    if (ext_str == ".json") {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::cout << "ERROR! can't open JSON file to write : (" << filename << ")" << std::endl;
            return;
        }
        ofs << std::setw(4) << json << std::endl;

    } else if (ext_str == ".dat" || ext_str == ".cbor") {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs.is_open()) {
            std::cout << "ERROR!! can't open DAT file to write : (" << filename << ")" << std::endl;
            return;
        }
        auto cbor = njson::to_cbor(json);
        ofs.write(reinterpret_cast<char *>(cbor.data()), cbor.size());

    } else {
        std::cout << "ERROR! not support file type to write : " << ext_str << "." << std::endl;
        return;
    }
}

}
