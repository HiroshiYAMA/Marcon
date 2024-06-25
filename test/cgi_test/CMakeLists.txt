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

# compile executable file
add_executable(server Server.cpp)
add_executable(client Client.cpp)

target_include_directories(server PUBLIC ${OPENSSL_INCLUDE_DIR} ${CPP_HTTPLIB_PATH})
target_include_directories(client PUBLIC ${OPENSSL_INCLUDE_DIR} ${CPP_HTTPLIB_PATH})
target_link_libraries(server ${OPENSSL_LIBRARIES} pthread)
target_link_libraries(client ${OPENSSL_LIBRARIES} pthread)
# target_compile_definitions(server HOGEHOGE)
