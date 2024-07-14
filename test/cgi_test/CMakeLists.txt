cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# require OpenSSL v3. If you need, set path of OpenSSL v3.
#set(ENV{OPENSSL_ROOT_DIR} /opt/openssl3)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set the project name and version
project(httpSample CXX)

# OpenSSL v3.
find_package(OpenSSL REQUIRED)
message("OpenSSL : ${OPENSSL_VERSION}")
message("OpenSSL include dirs: ${OPENSSL_INCLUDE_DIR}")
message("OpenSSL library dirs: ${OPENSSL_LIBRARY_DIR}")
message("OpenSSL libraries: ${OPENSSL_CRYPTO_LIBRARIES}")
message("OpenSSL libraries: ${OPENSSL_SSL_LIBRARIES}")
message("OpenSSL libraries: ${OPENSSL_LIBRARIES}")

# cpp-httplib.
set(CPP_HTTPLIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../external/cpp-httplib")

# C++ REST SDK.
find_package(cpprestsdk REQUIRED)

# JSON.
set(JSON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../external/json/single_include/nlohmann")
add_definitions("-DJSON_USE_IMPLICIT_CONVERSIONS=0")

set(LOCAL_INCLUDE_PATH ${CPP_HTTPLIB_PATH} ${JSON_PATH})

# compile executable file
add_executable(server Server.cpp)
add_executable(client Client.cpp)
add_executable(cpprest_client cpprest_client.cpp)

target_include_directories(server PUBLIC ${OPENSSL_INCLUDE_DIR} ${LOCAL_INCLUDE_PATH})
target_include_directories(client PUBLIC ${OPENSSL_INCLUDE_DIR} ${LOCAL_INCLUDE_PATH})
target_link_libraries(server ${OPENSSL_LIBRARIES} pthread)
target_link_libraries(client ${OPENSSL_LIBRARIES} pthread)
# target_compile_definitions(server HOGEHOGE)

target_include_directories(cpprest_client PUBLIC ${JSON_PATH})
target_link_libraries(cpprest_client cpprestsdk::cpprest)
