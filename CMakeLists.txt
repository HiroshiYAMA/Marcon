cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(marcon LANGUAGES CXX)
set(target "${PROJECT_NAME}")

set(CMAKE_INSTALL_PREFIX "../install")
set(exe_install_path ".")

set(SRC_DIR "src")
set(EXTERNAL_DIR "external")
set(GLFW_DIR "${EXTERNAL_DIR}/glfw")
set(IMGUI_DIR "${EXTERNAL_DIR}/imgui")

add_executable(${target}
    ${SRC_DIR}/marcon.cpp
    ${SRC_DIR}/gui_utils.cpp
    ${SRC_DIR}/CGI.cpp
    ${SRC_DIR}/CGI_cmd_parameter.cpp
    ${SRC_DIR}/search_ipadr.cpp

    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

set_target_properties(${target} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

if(APPLE)
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "@executable_path"
        INSTALL_RPATH "@executable_path"
    )
else(APPLE)
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif(APPLE)

### Linux specific configuration ###
if(UNIX AND NOT APPLE)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
            # Must use std::experimental namespace if older than GCC8
            message("[${PROJECT_NAME}] GCC version less than 8. Using std::experimental namespace.")
            target_compile_definitions(${target} PRIVATE USE_EXPERIMENTAL_FS)
        endif()

        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
            # Must explicitly link separate std::filesystem if older than GCC9
            message("[${PROJECT_NAME}] GCC version less than 9. Explicitly linking separate std::filesystem library.")
            target_link_libraries(${target} stdc++fs)
        endif()
    endif()
endif(UNIX AND NOT APPLE)

# # SDL2.
# if(NOT WIN32)
#     find_package(PkgConfig REQUIRED)
#     pkg_check_modules(SDL2 REQUIRED sdl2)
# else(NOT WIN32)
#     set(SDL2_INCLUDE_DIRS "c:/SDL2/include")
#     set(SDL2_LIBRARY_DIRS "c:/SDL2/lib/x64")
#     set(SDL2_LIBRARIES "${SDL2_LIBRARY_DIRS}/SDL2.lib" "${SDL2_LIBRARY_DIRS}/SDL2main.lib")
# endif(NOT WIN32)
# message("SDL2 : ${SDL2_VERSION}")
# message("SDL2 include dirs: ${SDL2_INCLUDE_DIRS}")
# message("SDL2 library dirs: ${SDL2_LIBRARY_DIRS}")
# message("SDL2 libraries: ${SDL2_LIBRARIES}")
# message("SDL2 cflags: ${SDL2_CFLAGS}")
# target_compile_definitions(${target} PUBLIC _REENTRANT _THREAD_SAFE)
# target_include_directories(${target} PUBLIC ${SDL2_INCLUDE_DIRS})
# target_link_libraries(${target} ${SDL2_LIBRARIES})
# target_link_directories(${target} PUBLIC ${SDL2_LIBRARY_DIRS})

# GLFW.
add_subdirectory(${GLFW_DIR})
target_include_directories(${target} PUBLIC ${GLFW_DIR}/include)
target_link_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/build/${GLFW_DIR}/src)
target_link_libraries(${target} libglfw3.a pthread)

# OpenCV.
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv4)
message("OpenCV : ${OpenCV_VERSION}")
message("OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message("OpenCV libraries: ${OpenCV_LIBRARIES}")
target_include_directories(${target} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${target} ${OpenCV_LIBRARIES})
target_compile_definitions(${target} PUBLIC GAPI_ENABLE)

if(APPLE)
    target_include_directories(${target} PUBLIC /usr/local/include)
    target_link_directories(${target} PUBLIC /usr/local/lib)
    target_link_libraries(${target} libGLEW.a)
    target_link_libraries(${target} "-framework OpenGL")
    target_link_libraries(${target} "-framework Cocoa")
    target_link_libraries(${target} "-framework IOKit")
endif(APPLE)

# ImGui.
set(IMGUI_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/${IMGUI_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/${IMGUI_DIR}/backends"
)
target_include_directories(${target} PUBLIC ${IMGUI_INCLUDE_DIRS})
if(NOT APPLE)
    target_link_libraries(${target} GL dl)
endif(APPLE)

# JSON.
set(JSON_INCLUDE_DIR "${EXTERNAL_DIR}/json/single_include/nlohmann")
target_compile_definitions(${target} PUBLIC "JSON_USE_IMPLICIT_CONVERSIONS=0")
target_include_directories(${target} PUBLIC ${JSON_INCLUDE_DIR})

# require OpenSSL v3. If you need, set path of OpenSSL v3.
# set(ENV{OPENSSL_ROOT_DIR} /opt/openssl3)

# OpenSSL v3.
find_package(OpenSSL REQUIRED)
message("OpenSSL : ${OPENSSL_VERSION}")
message("OpenSSL include dirs: ${OPENSSL_INCLUDE_DIR}")
message("OpenSSL library dirs: ${OPENSSL_LIBRARY_DIR}")
message("OpenSSL libraries: ${OPENSSL_CRYPTO_LIBRARIES}")
message("OpenSSL libraries: ${OPENSSL_SSL_LIBRARIES}")
message("OpenSSL libraries: ${OPENSSL_LIBRARIES}")
target_include_directories(${target} PUBLIC ${OPENSSL_INCLUDE_DIR})
target_link_libraries(${target} ${OPENSSL_LIBRARIES} z)

# # cpp-httplib.
# set(CPP_HTTPLIB_INCLUDE_DIR "${EXTERNAL_DIR}/cpp-httplib")
# target_include_directories(${target} PUBLIC ${CPP_HTTPLIB_INCLUDE_DIR})

# C++ Requests.
set(BUILD_LIBCURL_DOCS OFF)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_CURL ON)
set(BUILD_STATIC_LIBS ON)
include(FetchContent)
FetchContent_Declare(cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git
                         GIT_TAG 3b15fa82ea74739b574d705fea44959b58142eb8) # Replace with your desired git commit from: https://github.com/libcpr/cpr/releases
FetchContent_MakeAvailable(cpr)
target_link_libraries(${target} cpr::cpr)

# using NVIDIA plugins of GStreamer.
# target_compile_definitions(${target} PUBLIC GST_NV)

# using NVIDIA Jetson.
# target_compile_definitions(${target} PUBLIC JETSON)

# using Apple VideoToolbox plugins of GStreamer.
# target_compile_definitions(${target} PUBLIC GST_APPLE)

## Install path defined in parent CMakeLists
install(TARGETS ${target} DESTINATION ${exe_install_path})
