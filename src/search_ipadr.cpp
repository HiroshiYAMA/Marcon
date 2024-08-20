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

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

#include <chrono>
#include <mutex>
#include <thread>
#include <signal.h>

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

#include "IpNetwork.h"

void init_search_ipadr()
{
#ifdef _MSC_VER
	{
		WSADATA wsa_data;
		if (WSAStartup(MAKEWORD(2, 0), &wsa_data) != 0) {
			std::cout << "ERROR!! WSAStartup." << std::endl;
			exit(EXIT_FAILURE);
		}

		auto wsa_cleanup = []() -> void {
			WSACleanup();
		};
		atexit(wsa_cleanup);
	}
#endif
}

std::vector<st_NetInfo> search_ipadr()
{
	auto bc_list = IpNetwork::get_broadcast_address_list(AF_INET);

	std::vector<st_NetInfo> ip_adr_list;

	for (auto &e : bc_list) {
		auto sa = *((sockaddr_in *)&e);
		char name[NI_MAXHOST];
		char service[NI_MAXSERV] = "52380";
		auto flags = NI_DGRAM | NI_NUMERICHOST | NI_NUMERICSERV;

		auto err = getnameinfo((sockaddr *)&sa, sizeof(sa), name, sizeof(name), NULL, 0, flags);
		if (err) continue;

		std::unique_ptr<IpNetwork::Connection> ipnet;
		ipnet = IpNetwork::UDP::Create(name, service, IpNetwork::Connection::em_Mode::SEND, true, "", 0, true);

		if (!ipnet) {
			LogError("ERROR!! create IP Network object.\n");
			continue;
		}

		auto net_info = IpNetwork::NetworkInfo::Create();
		auto pkt = net_info->pack();
		ipnet->send(pkt);

		constexpr auto COUNT = 30;
		constexpr auto SEARCH_TIME = 1'000.0;	// [msec].
		TinyTimer tt;
		StopWatch sw;
		auto time_over = [](auto sw) -> bool {
			sw.stop();
			auto dt_ms = sw.duration();
			return (dt_ms > (SEARCH_TIME * 1.1));
		};

		int cnt = 0;
		while (cnt++ < COUNT) {
			while (ipnet->poll()) {
				if (ipnet->is_empty_receive_queue()) break;

				while (auto pkt = ipnet->receive()) {
					net_info->unpack(*pkt);

					st_NetInfo ni;
					ni.ipadr = net_info->get_netinfo_ip_address();
					ni.nickname = net_info->get_netinfo_name();
					ni.mac = net_info->get_netinfo_mac_address();
					ip_adr_list.push_back(ni);

					if (time_over(sw)) break;
				}

				if (time_over(sw)) break;
			}

			tt.wait1period(SEARCH_TIME / COUNT);
		}
	}
	std::sort(ip_adr_list.begin(), ip_adr_list.end());

	std::cout
		<< std::endl
		<< "IP address list :" << std::endl
		<< "-----------------------------------------------" << std::endl;
	for (auto &e : ip_adr_list) {
		njson json = e;
		std::cout << std::setw(4) << json << std::endl;
	}
	std::cout << "-----------------------------------------------" << std::endl;

	return ip_adr_list;
}
