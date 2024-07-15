cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set the project name and version
project(httpSample CXX)

# C++ Requests.
set(BUILD_LIBCURL_DOCS OFF)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_CURL ON)
set(BUILD_STATIC_LIBS ON)
include(FetchContent)
FetchContent_Declare(cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git
                         GIT_TAG 3b15fa82ea74739b574d705fea44959b58142eb8) # Replace with your desired git commit from: https://github.com/libcpr/cpr/releases
FetchContent_MakeAvailable(cpr)

# JSON.
set(JSON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../external/json/single_include/nlohmann")
add_definitions("-DJSON_USE_IMPLICIT_CONVERSIONS=0")

# compile executable file
add_executable(cpr_client cpr_client.cpp)

target_include_directories(cpr_client PUBLIC ${JSON_PATH})
target_link_libraries(cpr_client cpr::cpr)
