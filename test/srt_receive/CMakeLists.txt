cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# set the project name and version
project(srt_receive LANGUAGES CXX)
set(target "${PROJECT_NAME}")

set(CMAKE_INSTALL_PREFIX "../install")
set(exe_install_path ".")

# compile executable file
add_executable(${target} srt_receive.cpp)

#target_compile_features(segnet_lanuch PUBLIC cxx_std_17)
set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
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

# OpenCV.
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv4)
message("OpenCV : ${OpenCV_VERSION}")
message("OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message("OpenCV libraries: ${OpenCV_LIBRARIES}")

target_include_directories(${target} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${target} ${OpenCV_LIBRARIES})

# using NVIDIA plugins of GStreamer.
# target_compile_definitions(${target} PUBLIC GST_NV)

## Install path defined in parent CMakeLists
install(TARGETS ${target} DESTINATION ${exe_install_path})
