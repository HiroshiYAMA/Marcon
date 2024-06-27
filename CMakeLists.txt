cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(marcon LANGUAGES CXX)
set(target "${PROJECT_NAME}")

set(CMAKE_INSTALL_PREFIX "../install")
set(exe_install_path ".")

set(SRC_DIR "src")
set(IMGUI_DIR "external/imgui")

add_executable(${target}
    ${SRC_DIR}/marcon.cpp

    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
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

# SDL2.
if(NOT WIN32)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SDL2 REQUIRED sdl2)
else(NOT WIN32)
    set(SDL2_INCLUDE_DIRS "c:/SDL2/include")
    set(SDL2_LIBRARY_DIRS "c:/SDL2/lib/x64")
    set(SDL2_LIBRARIES "${SDL2_LIBRARY_DIRS}/SDL2.lib" "${SDL2_LIBRARY_DIRS}/SDL2main.lib")
endif(NOT WIN32)
message("SDL2 : ${SDL2_VERSION}")
message("SDL2 include dirs: ${SDL2_INCLUDE_DIRS}")
message("SDL2 library dirs: ${SDL2_LIBRARY_DIRS}")
message("SDL2 libraries: ${SDL2_LIBRARIES}")
message("SDL2 cflags: ${SDL2_CFLAGS}")
target_compile_definitions(${target} PUBLIC _REENTRANT _THREAD_SAFE)
target_include_directories(${target} PUBLIC ${SDL2_INCLUDE_DIRS})
target_link_libraries(${target} ${SDL2_LIBRARIES})
target_link_directories(${target} PUBLIC ${SDL2_LIBRARY_DIRS})

# OpenCV.
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv4)
message("OpenCV : ${OpenCV_VERSION}")
message("OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message("OpenCV libraries: ${OpenCV_LIBRARIES}")
target_include_directories(${target} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${target} ${OpenCV_LIBRARIES})

# ImGui.
set(IMGUI_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends"
)
target_include_directories(${target} PUBLIC ${IMGUI_INCLUDE_DIRS})
target_link_libraries(${target} GL dl)

# using NVIDIA plugins of GStreamer.
# target_compile_definitions(${target} PUBLIC GST_NV)

## Install path defined in parent CMakeLists
install(TARGETS ${target} DESTINATION ${exe_install_path})
