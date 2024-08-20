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

#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>
#include <list>
#include <deque>
#include <queue>
#include <tuple>
#include <map>
#include <type_traits>

#include <chrono>
#include <mutex>
#include <thread>

#ifdef USE_SRT
#include <srt.h>
#else
#ifdef _MSC_VER
#include <WinSock2.h>
#include <Ws2tcpip.h>
#include <iphlpapi.h>
#include <sys/types.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#endif
#endif
#ifdef __linux__
#include <sys/epoll.h>
#endif

#include "json.hpp"
using njson = nlohmann::json;



#ifdef _MSC_VER
#if 0
inline auto w2s = [](std::wstring w) -> std::string {
	size_t l_c;
	auto l = w.size() * 2;
	char *c = (char *)malloc(sizeof(char) * l);
	std::string s;
	if (c) {
		wcstombs_s(&l_c, c, l, w.c_str(), l);
		s = c;
		free(c);
	} else {
		s = "";
	}
	return s;
};
#else
#include "AtlBase.h"    // unicode(UTF-16) から ShiftJIS に変換用
inline auto w2s = [](std::wstring w) -> std::string {
	USES_CONVERSION;
    auto c = W2A(w.c_str());
	std::string s;
    if (c) {
        s = c;
    } else {
        s = "";
    }
	return s;
};
#endif

static std::string get_mac_address(PIP_ADAPTER_ADDRESSES adpt)
{
    auto mac_adr_ary = adpt->PhysicalAddress;
    auto mac_adr_size = adpt->PhysicalAddressLength;

    std::stringstream ss;
    ss << std::uppercase << std::hex;
    ss << std::setw(2) << std::setfill('0') << (int)mac_adr_ary[0];
    for (size_t i = 1; i < mac_adr_size; i++) {
        ss << "-" << std::setw(2) << std::setfill('0') << (int)mac_adr_ary[i];
    }
    std::string str = ss.str();

    return str;
}

static sockaddr_storage get_broadcast_address(PIP_ADAPTER_UNICAST_ADDRESS addr_unicast)
{
    sockaddr_storage sa{};

    auto sock_addr = addr_unicast->Address.lpSockaddr;
    auto mask_bits = addr_unicast->OnLinkPrefixLength;
    auto family = sock_addr->sa_family;
    if (family == AF_INET) {
        uint32_t mask = (0xFFFF'FFFF << (32 - mask_bits));
        auto sa_in = *((sockaddr_in *)sock_addr);
        uint32_t adr = ntohl(sa_in.sin_addr.S_un.S_addr);
        adr |= ~mask;
        sa_in.sin_addr.S_un.S_addr = htonl(adr);
        *(sockaddr_in *)&sa = sa_in;

    } else if (family == AF_INET6) {
        ;
    }

    return sa;
}
#endif

static std::string get_ipadr_port_str(sockaddr *addr)
{
    char adr[256];
    uint16_t port;

    switch (addr->sa_family) {
    case AF_INET:
        {
            auto addr_in = (sockaddr_in *)addr;
            inet_ntop(AF_INET, &addr_in->sin_addr, adr, sizeof(adr));
            port = ntohs(addr_in->sin_port);
        }
        break;
    case AF_INET6:
        {
            auto addr_in6 = (sockaddr_in6 *)addr;
            inet_ntop(AF_INET6, &addr_in6->sin6_addr, adr, sizeof(adr));
            port = ntohs(addr_in6->sin6_port);
        }
        break;
    default:
        adr[0] = '\0';
        port = 0;
    }

    std::string str = std::string{adr} + ":" + std::to_string(port);

    return str;
}

inline auto print_bytestream = [](const auto &data, int cols = 16) -> void {
    int str_len = cols * 3 + 12;
    std::cout << std::string(str_len, '-') << std::endl;

    for (auto i = 0; i < data.size(); i++) {
        if ((i % cols) == 0) std::cout << std::hex << std::setfill('0') << std::setw(8) << i << " :";
        if ((i % cols) == (cols / 2)) std::cout << " |";

        std::cout << " " << std::hex << std::setfill('0') << std::setw(2) << (int)data[i];

        if ((i % cols) == (cols - 1)) std::cout << std::endl;
    }
    std::cout << std::dec << std::setfill(' ') << std::endl;

    std::cout << std::string(str_len, '-') << std::endl;
};

inline auto set_fio_blocking_mode = [](auto m_sock) -> bool {
#if defined(_WIN32)
    unsigned long ulyes = 0;
    return !(ioctlsocket(m_sock, FIONBIO, &ulyes) == SOCKET_ERROR);
#else
    return !(fcntl(m_sock, F_SETFL, fcntl(m_sock, F_GETFL, 0) & ~O_NONBLOCK) < 0);
#endif
};

inline auto set_fio_non_blocking_mode = [](auto m_sock) -> bool {
#if defined(_WIN32)
    unsigned long ulyes = 1;
    return !(ioctlsocket(m_sock, FIONBIO, &ulyes) == SOCKET_ERROR);
#else
    return !(fcntl(m_sock, F_SETFL, fcntl(m_sock, F_GETFL, 0) | O_NONBLOCK) < 0);
#endif
};
