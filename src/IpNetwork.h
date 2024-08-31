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

#include "common_utils.h"
#include "common_inet.h"
#include <regex>
#ifdef _MSC_VER
using sa_family_t = ADDRESS_FAMILY;
#else
#include <ifaddrs.h>
#include <net/if.h>
#include <arpa/inet.h>
#endif

#define LogInfo printf
#define LogError printf
#define LogSuccess printf
#define LogVerbose printf
#define LogDebug printf



namespace IpNetwork {
using packet_type = std::vector<uint8_t>;

enum class em_DataType : int {
	NETWORK_INFO,
	INVALID = -1,
};

static const std::unordered_map<em_DataType, const char *> text_list_data_type = {
	{ em_DataType::NETWORK_INFO, "Network Infomation" },
};
static const char *get_data_type_str(em_DataType data_type)
{
	auto itr = text_list_data_type.find(data_type);
	auto err_value = std::make_pair(em_DataType::INVALID, "invalid data type");
	auto [key, value] = (itr != text_list_data_type.end() ? decltype(err_value){*itr} : err_value);
	return value;
}

static bool check_multicast(const addrinfo *ai)
{
	bool ret = false;

	switch (ai->ai_family) {
	case AF_INET:
		{
			auto adr = ((sockaddr_in *)ai->ai_addr)->sin_addr.s_addr;
			adr = ntohl(adr);
			ret = IN_MULTICAST(adr);
		}
		break;
	case AF_INET6:
		{
			auto adr = ((sockaddr_in6 *)ai->ai_addr)->sin6_addr;
#ifdef __linux__
			ret = IN6_IS_ADDR_MULTICAST(adr.s6_addr);
#else
			ret = IN6_IS_ADDR_MULTICAST(&adr);
#endif
		}
		break;
	default:
		std::cout << "not AF_INET/AF_INET6" << std::endl;
	}

	return ret;
}

static bool check_broadcast(const addrinfo *ai)
{
	bool ret = false;

	switch (ai->ai_family) {
	case AF_INET:
		{
			auto adr = ((sockaddr_in *)ai->ai_addr)->sin_addr.s_addr;
			adr = ntohl(adr);
			ret = (adr == INADDR_BROADCAST);
		}
		break;
	case AF_INET6:
		{
			auto adr = ((sockaddr_in6 *)ai->ai_addr)->sin6_addr;
#ifdef __linux__
			ret = IN6_IS_ADDR_MC_NODELOCAL(adr.s6_addr)
				|| IN6_IS_ADDR_MC_LINKLOCAL(adr.s6_addr)
				|| IN6_IS_ADDR_MC_SITELOCAL(adr.s6_addr)
				|| IN6_IS_ADDR_MC_ORGLOCAL(adr.s6_addr)
				|| IN6_IS_ADDR_MC_GLOBAL(adr.s6_addr)
				;
#else
			ret = IN6_IS_ADDR_MULTICAST(&adr);
#endif
		}
		break;
	default:
		std::cout << "not AF_INET/AF_INET6" << std::endl;
	}

	return ret;
}

union inet_addr
{
	in_addr adr4;
	in6_addr adr6 = IN6ADDR_ANY_INIT;
};

#ifdef _MSC_VER
static std::list<std::string> get_netif_name_list(sa_family_t net_family, unsigned int net_flags = IfOperStatusUp)
{
	auto name_list = std::list<std::string>{};

	PIP_ADAPTER_ADDRESSES ifa0 = NULL;
	DWORD ifa0_size;
	if (GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, NULL, &ifa0_size) != ERROR_BUFFER_OVERFLOW) {
		return {};
	}
	ifa0 = (PIP_ADAPTER_ADDRESSES)calloc(ifa0_size, sizeof(BYTE));
	if (!ifa0) return {};
	if (GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, ifa0, &ifa0_size) != ERROR_SUCCESS) {
		free(ifa0);
		return {};
	}

	auto ifa = ifa0;
	while (ifa) {
		// auto name = std::string{w2s(std::wstring{ifa->FriendlyName})};
		auto mac_adr = get_mac_address(ifa);

		auto addr_unicast = ifa->FirstUnicastAddress;
		auto op_stat = ifa->OperStatus;
		while (addr_unicast && (op_stat == net_flags)) {
			auto sock_addr = addr_unicast->Address.lpSockaddr;
			auto family = sock_addr->sa_family;
			if (family == net_family) {
				name_list.push_back(mac_adr);
			}
			addr_unicast = addr_unicast->Next;
		}

		ifa = ifa->Next;
	}

	free(ifa0);

	return name_list;
}

static std::tuple<inet_addr, unsigned int> get_netif_stat(const std::string &netif, sa_family_t net_family)
{
	auto ret = std::make_tuple(inet_addr{}, 0);

	PIP_ADAPTER_ADDRESSES ifa0 = NULL;
	DWORD ifa0_size;
	if (GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, NULL, &ifa0_size) != ERROR_BUFFER_OVERFLOW) {
		return ret;
	}
	ifa0 = (PIP_ADAPTER_ADDRESSES)calloc(ifa0_size, sizeof(BYTE));
	if (!ifa0) return ret;
	if (GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, ifa0, &ifa0_size) != ERROR_SUCCESS) {
		free(ifa0);
		return ret;
	}

	auto ifa = ifa0;
	while (ifa) {
		// auto name = std::string{w2s(std::wstring{ifa->FriendlyName})};
		// if (name != netif) { ifa = ifa->Next; continue; }
		auto mac_adr = get_mac_address(ifa);
		if (mac_adr != netif) { ifa = ifa->Next; continue; }

		auto addr_unicast = ifa->FirstUnicastAddress;
		while (addr_unicast) {
			auto sock_addr = addr_unicast->Address.lpSockaddr;
			auto family = sock_addr->sa_family;
			if (family == net_family) {
				inet_addr adr = {};
				IF_INDEX index;
				switch (family) {
				case AF_INET:
					adr.adr4 = ((sockaddr_in *)sock_addr)->sin_addr;
					index = ifa->IfIndex;
					break;
				case AF_INET6:
					adr.adr6 = ((sockaddr_in6 *)sock_addr)->sin6_addr;
					index = ifa->Ipv6IfIndex;
					break;
				default:
					;
				}
				ret = { adr, index };
				break;
			} else {
				addr_unicast = addr_unicast->Next;
				continue;
			}
		}
		break;
	}

	free(ifa0);

	return ret;
}

static sockaddr_storage get_broadcast_address(const std::string &netif, sa_family_t net_family)
{
	auto sa = sockaddr_storage{};

	PIP_ADAPTER_ADDRESSES ifa0 = NULL;
	DWORD ifa0_size;
	if (GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, NULL, &ifa0_size) != ERROR_BUFFER_OVERFLOW) {
		return sa;
	}
	ifa0 = (PIP_ADAPTER_ADDRESSES)calloc(ifa0_size, sizeof(BYTE));
	if (!ifa0) return sa;
	if (GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, ifa0, &ifa0_size) != ERROR_SUCCESS) {
		free(ifa0);
		return sa;
	}

	auto ifa = ifa0;
	while (ifa) {
		auto addr_unicast = ifa->FirstUnicastAddress;
		auto op_stat = ifa->OperStatus;
		auto mac_adr = get_mac_address(ifa);
		while ((netif == mac_adr) && addr_unicast && (op_stat == IfOperStatusUp)) {
			auto sock_addr = addr_unicast->Address.lpSockaddr;
			auto family = sock_addr->sa_family;
			if (family == net_family) {
				sa = get_broadcast_address(addr_unicast);
				break;
			}
			addr_unicast = addr_unicast->Next;
		}

		ifa = ifa->Next;
	}

	free(ifa0);

	return sa;
}

static std::list<sockaddr_storage> get_broadcast_address_list(sa_family_t net_family)
{
	auto sa_list = std::list<sockaddr_storage>{};

	PIP_ADAPTER_ADDRESSES ifa0 = NULL;
	DWORD ifa0_size;
	if (GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, NULL, &ifa0_size) != ERROR_BUFFER_OVERFLOW) {
		return sa_list;
	}
	ifa0 = (PIP_ADAPTER_ADDRESSES)calloc(ifa0_size, sizeof(BYTE));
	if (!ifa0) return sa_list;
	if (GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, ifa0, &ifa0_size) != ERROR_SUCCESS) {
		free(ifa0);
		return sa_list;
	}

	auto ifa = ifa0;
	while (ifa) {
		auto addr_unicast = ifa->FirstUnicastAddress;
		auto op_stat = ifa->OperStatus;
		while (addr_unicast && (op_stat == IfOperStatusUp)) {
			auto sock_addr = addr_unicast->Address.lpSockaddr;
			auto family = sock_addr->sa_family;
			if (family == net_family) {
				sockaddr_storage sa = {};
				switch (family) {
				case AF_INET:
					sa = get_broadcast_address(addr_unicast);
					break;
				case AF_INET6:
					;
					break;
				default:
					;
				}
				sa_list.push_back(sa);
			}
			addr_unicast = addr_unicast->Next;
		}

		ifa = ifa->Next;
	}

	free(ifa0);

	return sa_list;
}
#else
static std::list<std::string> get_netif_name_list(sa_family_t net_family, unsigned int net_flags = 0)
{
	auto name_list = std::list<std::string>{};

	ifaddrs *ifa0 = nullptr;
	if (getifaddrs(&ifa0) == -1) {
		if (ifa0) freeifaddrs(ifa0);
		return name_list;
	}

	auto ifa = ifa0;
	while (ifa) {
		if (!ifa->ifa_addr) { ifa = ifa->ifa_next; continue; }

		auto name = std::string{ifa->ifa_name};
		[[maybe_unused]] auto index = if_nametoindex(name.c_str());
		auto family = ifa->ifa_addr->sa_family;
		auto flags = ifa->ifa_flags;

		if ((family == net_family) && ((flags & net_flags) == net_flags)) {
			name_list.push_back(name);
		}

		ifa = ifa->ifa_next;
	}

	if (ifa0) freeifaddrs(ifa0);

	return name_list;
}

static std::tuple<inet_addr, unsigned int> get_netif_stat(const std::string &netif, sa_family_t net_family)
{
	auto ret = std::make_tuple(inet_addr{}, 0);

	ifaddrs *ifa0 = nullptr;
	if (getifaddrs(&ifa0) == -1) {
		if (ifa0) freeifaddrs(ifa0);
		return ret;
	}

	auto ifa = ifa0;
	while (ifa) {
		if (!ifa->ifa_addr) { ifa = ifa->ifa_next; continue; }

		auto name = std::string{ifa->ifa_name};
		auto index = if_nametoindex(name.c_str());
		auto family = ifa->ifa_addr->sa_family;

		if (name == netif && family == net_family) {
			inet_addr adr = {};
			switch (family) {
			case AF_INET:
				adr.adr4 = ((sockaddr_in *)ifa->ifa_addr)->sin_addr;
				break;
			case AF_INET6:
				adr.adr6 = ((sockaddr_in6 *)ifa->ifa_addr)->sin6_addr;
				break;
			default:
				;
			}
			ret = { adr, index };
			break;
		}

		ifa = ifa->ifa_next;
	}

	if (ifa0) freeifaddrs(ifa0);

	return ret;
}

static sockaddr_storage get_broadcast_address(const std::string &netif, sa_family_t net_family)
{
	auto sa = sockaddr_storage{};

	ifaddrs *ifa0 = nullptr;
	if (getifaddrs(&ifa0) == -1) {
		if (ifa0) freeifaddrs(ifa0);
		return sa;
	}

	auto ifa = ifa0;
	while (ifa) {
		if (!ifa->ifa_addr) { ifa = ifa->ifa_next; continue; }

		auto name = std::string{ifa->ifa_name};
		[[maybe_unused]] auto index = if_nametoindex(name.c_str());
		auto family = ifa->ifa_addr->sa_family;
		auto flags = ifa->ifa_flags;
		constexpr auto flags_bit = IFF_BROADCAST | IFF_UP | IFF_RUNNING;

		if ((name == netif) && (family == net_family) && ((flags & flags_bit)) == flags_bit) {
			switch (family) {
			case AF_INET:
#ifdef __APPLE__
				*(sockaddr_in *)&sa = *((sockaddr_in *)ifa->ifa_dstaddr);
#else
				*(sockaddr_in *)&sa = *((sockaddr_in *)ifa->ifa_ifu.ifu_broadaddr);
#endif
				break;
			case AF_INET6:
#ifdef __APPLE__
				*(sockaddr_in6 *)&sa = *((sockaddr_in6 *)ifa->ifa_dstaddr);
#else
				*(sockaddr_in6 *)&sa = *((sockaddr_in6 *)ifa->ifa_ifu.ifu_broadaddr);
#endif
				break;
			default:
				;
			}
			break;
		}

		ifa = ifa->ifa_next;
	}

	if (ifa0) freeifaddrs(ifa0);

	return sa;
}

static std::list<sockaddr_storage> get_broadcast_address_list(sa_family_t net_family)
{
	auto sa_list = std::list<sockaddr_storage>{};

	ifaddrs *ifa0 = nullptr;
	if (getifaddrs(&ifa0) == -1) {
		if (ifa0) freeifaddrs(ifa0);
		return sa_list;
	}

	auto ifa = ifa0;
	while (ifa) {
		if (!ifa->ifa_addr) { ifa = ifa->ifa_next; continue; }

		[[maybe_unused]] auto name = std::string{ifa->ifa_name};
		[[maybe_unused]] auto index = if_nametoindex(name.c_str());
		auto family = ifa->ifa_addr->sa_family;
		auto flags = ifa->ifa_flags;
		constexpr auto flags_mask = IFF_BROADCAST | IFF_UP | IFF_RUNNING;

		if ((family == net_family) && ((flags & flags_mask) == flags_mask)) {
			sockaddr_storage sa = {};
			switch (family) {
			case AF_INET:
#ifdef __APPLE__
				*(sockaddr_in *)&sa = *((sockaddr_in *)ifa->ifa_dstaddr);
#else
				*(sockaddr_in *)&sa = *((sockaddr_in *)ifa->ifa_ifu.ifu_broadaddr);
#endif
				break;
			case AF_INET6:
#ifdef __APPLE__
				*(sockaddr_in6 *)&sa = *((sockaddr_in6 *)ifa->ifa_dstaddr);
#else
				*(sockaddr_in6 *)&sa = *((sockaddr_in6 *)ifa->ifa_ifu.ifu_broadaddr);
#endif
				break;
			default:
				;
			}
			sa_list.push_back(sa);
		}

		ifa = ifa->ifa_next;
	}

	if (ifa0) freeifaddrs(ifa0);

	return sa_list;
}
#endif

static bool set_multicast_caller(int sock, const std::string &mc_if, int mc_ttl, sa_family_t family)
{
	int ret = 0;

	auto [adr, idx] = get_netif_stat(mc_if, family);
	switch (family) {
	case AF_INET:
		{
			auto adr4 = adr.adr4;
			uint8_t ttl = mc_ttl;
			ret += setsockopt(sock, IPPROTO_IP, IP_MULTICAST_IF, (const char *)&adr4, sizeof(adr4));
			ret += setsockopt(sock, IPPROTO_IP, IP_MULTICAST_TTL, (const char *)&ttl, sizeof(ttl));
		}
		break;
	case AF_INET6:
		ret += setsockopt(sock, IPPROTO_IPV6, IPV6_MULTICAST_IF, (const char *)&idx, sizeof(idx));
		ret += setsockopt(sock, IPPROTO_IPV6, IPV6_MULTICAST_HOPS, (const char *)&mc_ttl, sizeof(mc_ttl));
		break;
	default:
		ret = -1;
	}

	return (ret == 0);
}

static bool multicast_join_group(int sock, group_req &gr, sa_family_t family)
{
	int ret = 0;

	switch (family) {
	case AF_INET:
		ret += setsockopt(sock, IPPROTO_IP, MCAST_JOIN_GROUP, (char *)&gr, sizeof(gr));
		break;
	case AF_INET6:
		ret += setsockopt(sock, IPPROTO_IPV6, MCAST_JOIN_GROUP, (char *)&gr, sizeof(gr));
		break;
	default:
		ret = -1;
	}

	return (ret == 0);
}

static bool multicast_leave_group(int sock, group_req &gr, sa_family_t family)
{
	int ret = 0;

	switch (family) {
	case AF_INET:
		ret += setsockopt(sock, IPPROTO_IP, MCAST_LEAVE_GROUP, (char *)&gr, sizeof(gr));
		break;
	case AF_INET6:
		ret += setsockopt(sock, IPPROTO_IPV6, MCAST_LEAVE_GROUP, (char *)&gr, sizeof(gr));
		break;
	default:
		ret = -1;
	}

	return (ret == 0);
}

static bool set_multicast_group(int sock, const std::string &mc_if, const addrinfo *ai, std::unique_ptr<group_req> &mc_gr)
{
	int ret = 0;

	auto family = ai->ai_family;
	group_req gr;
	memset(&gr, 0, sizeof(gr));
	auto [adr, idx] = get_netif_stat(mc_if, family);
	gr.gr_interface = idx;
	memcpy(&(gr.gr_group), ai->ai_addr, ai->ai_addrlen);
	if (!multicast_join_group(sock, gr, family)) {
		return false;
	}

	mc_gr = std::make_unique<group_req>();
	if (!mc_gr) {
		multicast_leave_group(sock, gr, family);
		return false;
	}
	memcpy(mc_gr.get(), &gr, sizeof(group_req));

	return (ret == 0);
}



class IData
{
public:
	virtual packet_type pack() = 0;
	virtual void unpack(const packet_type &pkt) = 0;
};



class Data : public IData
{
protected:
	static constexpr uint8_t START_MARK = 0x02;
	static constexpr uint8_t END_MARK = 0x03;
	static constexpr uint8_t DELIMITER_MARK = 0xFF;
	static constexpr uint8_t PARAMETER_MARK = ':';

	static bool check_data(const packet_type &data)
	{
		if (data.size() < 3) return false;

		bool is_valid = (data.front() == START_MARK);
		is_valid &= (data.back() == END_MARK);
		is_valid &= (*(data.end() - 2) ==  DELIMITER_MARK);

		return is_valid;
	}

	static std::tuple<std::string, std::string> get_param_value(const std::string &str)
	{
		std::string param;
		std::string value;

		auto pos = str.find_first_of(PARAMETER_MARK);
		if (pos != std::string::npos) {
			param = str.substr(0, pos);
			value = str.substr(pos + 1);
		}

		return { param, value };
	}

public:
	Data() {}

	virtual ~Data() {}
};



class NetworkInfoRequest : public Data
{
private:
	static constexpr auto PARAM = "ENQ";
	static constexpr auto VALUE = "network";

	std::string param;
	std::string value;

protected:

public:
	packet_type pack() override
	{
		packet_type pkt;

		pkt.push_back(START_MARK);

		const std::string str = std::string{PARAM} + (char)PARAMETER_MARK + VALUE;
		pkt.insert(pkt.end(), str.cbegin(), str.cend());
		pkt.push_back(DELIMITER_MARK);

		pkt.push_back(END_MARK);

		return pkt;
	}

	void unpack(const packet_type &pkt) override
	{
		if (!check_data(pkt)) return;

		std::string str(pkt.begin() + 1, pkt.end() - 2);
		std::stringstream ss(str);
		std::vector<std::string> param_list;
		while (std::getline(ss, str, (char)DELIMITER_MARK)) {
			if (!str.empty()) param_list.push_back(str);
		}

		if (param_list.size() != 1) return;

		str = param_list[0];
		auto [p, v] = get_param_value(str);
		if (p == "" && v == "") return;

		param = p;
		value = v;

		std::cout << "[RECEIVE NetworkInfoRequest packet] size = " << pkt.size()
			<< "param = " << param
			<< "value = " << value
			<< std::endl;
		// print_bytestream(pkt);
	}

	static std::unique_ptr<NetworkInfoRequest> Create()
	{
		auto ptr = std::make_unique<NetworkInfoRequest>();
		if (!ptr) {
			LogError("Can't create NetworkInfoRequest.\n");
			return nullptr;
		}

		LogSuccess("create NetworkInfoRequest\n");

		return std::move(ptr);
	}

	NetworkInfoRequest() {}

	virtual ~NetworkInfoRequest() {}
};

class NetworkInfoResponse : public Data
{
private:
	static constexpr auto PARAM_MAC_ADDRESS = "MAC";
	static constexpr auto PARAM_INFOMATION = "INFO";
	static constexpr auto PARAM_MODEL = "MODEL";
	static constexpr auto PARAM_SOFT_VERSION = "SOFTVERSION";
	static constexpr auto PARAM_IP_ADDRESS = "IPADR";
	static constexpr auto PARAM_IP_MASK = "MASK";
	static constexpr auto PARAM_GATEWAY = "GATEWAY";
	static constexpr auto PARAM_NAME = "NAME";
	static constexpr auto PARAM_WRITE_ENABLE = "WRITE";

	std::map<std::string, std::string> netinfo_list = {};

protected:

public:
	packet_type pack() override
	{
		packet_type pkt;

		pkt.push_back(START_MARK);

		const std::vector<std::string> param_list = {
			PARAM_MAC_ADDRESS,
			PARAM_INFOMATION,
			PARAM_MODEL,
			PARAM_SOFT_VERSION,
			PARAM_IP_ADDRESS,
			PARAM_IP_MASK,
			PARAM_GATEWAY,
			PARAM_NAME,
			PARAM_WRITE_ENABLE,
		};
		for (const auto &param : param_list) {
			const auto &val = netinfo_list[param];
			const std::string str = param + (char)PARAMETER_MARK + val;
			pkt.insert(pkt.end(), str.cbegin(), str.cend());
			pkt.push_back(DELIMITER_MARK);
		}

		pkt.push_back(END_MARK);

		return pkt;
	}

	void unpack(const packet_type &pkt) override
	{
		if (!check_data(pkt)) return;

		std::string str(pkt.begin() + 1, pkt.end() - 2);
		std::stringstream ss(str);
		std::vector<std::string> param_list;
		while (std::getline(ss, str, (char)DELIMITER_MARK)) {
			if (!str.empty()) param_list.push_back(str);
		}

		if (param_list.size() == 0) return;

		for (const auto &e : param_list) {
			auto [param, value] = get_param_value(e);
			if (param == "" && value == "") continue;

			netinfo_list[param] = value;
		}

		njson json(netinfo_list);
		std::cout << "[RECEIVE NetworkInfoResponse packet] size = " << pkt.size()
			<< std::endl
			<< std::setw(4) << json
			<< std::endl;
		// print_bytestream(pkt);
	}
	
	std::string get_netinfo(const char *key) const
	{
		std::string str;

		try {
			str = netinfo_list.at(key);
		} catch(std::out_of_range& e) {
			str = "";
		}
		
		return str;
	}
	std::string get_netinfo_mac_address() const { return get_netinfo(PARAM_MAC_ADDRESS); }
	std::string get_netinfo_infomation() const { return get_netinfo(PARAM_INFOMATION); }
	std::string get_netinfo_model() const { return get_netinfo(PARAM_MODEL); }
	std::string get_netinfo_soft_version() const { return get_netinfo(PARAM_SOFT_VERSION); }
	std::string get_netinfo_ip_address() const { return get_netinfo(PARAM_IP_ADDRESS); }
	std::string get_netinfo_ip_mask() const { return get_netinfo(PARAM_IP_MASK); }
	std::string get_netinfo_gateway() const { return get_netinfo(PARAM_GATEWAY); }
	std::string get_netinfo_name() const { return get_netinfo(PARAM_NAME); }
	std::string get_netinfo_write_enable() const { return get_netinfo(PARAM_WRITE_ENABLE); }

	static std::unique_ptr<NetworkInfoResponse> Create()
	{
		auto ptr = std::make_unique<NetworkInfoResponse>();
		if (!ptr) {
			LogError("Can't create NetworkInfoResponse.\n");
			return nullptr;
		}

		LogSuccess("create NetworkInfoResponse\n");

		return std::move(ptr);
	}

	NetworkInfoResponse() {}

	virtual ~NetworkInfoResponse() {}
};

class NetworkInfo : public Data
{
private:
	std::unique_ptr<NetworkInfoRequest> request = nullptr;
	std::unique_ptr<NetworkInfoResponse> response = nullptr;

protected:

public:
	packet_type pack() override
	{
		auto pkt = request->pack();

		return pkt;
	}

	void unpack(const packet_type &pkt) override
	{
		response->unpack(pkt);
	}

	std::string get_netinfo_mac_address() const { return response->get_netinfo_mac_address(); }
	std::string get_netinfo_infomation() const { return response->get_netinfo_infomation(); }
	std::string get_netinfo_model() const { return response->get_netinfo_model(); }
	std::string get_netinfo_soft_version() const { return response->get_netinfo_soft_version(); }
	std::string get_netinfo_ip_address() const { return response->get_netinfo_ip_address(); }
	std::string get_netinfo_ip_mask() const { return response->get_netinfo_ip_mask(); }
	std::string get_netinfo_gateway() const { return response->get_netinfo_gateway(); }
	std::string get_netinfo_name() const { return response->get_netinfo_name();; }
	std::string get_netinfo_write_enable() const { return response->get_netinfo_write_enable(); }

	static std::unique_ptr<NetworkInfo> Create()
	{
		auto ptr = std::make_unique<NetworkInfo>();
		if (!ptr) {
			LogError("Can't create NetworkInfo.\n");
			return nullptr;
		}
		if (!ptr->request || !ptr->response) {
			LogError("Can't create request/response in NetworkInfo.\n");
			return nullptr;
		}

		LogSuccess("create NetworkInfo\n");

		return std::move(ptr);
	}

	NetworkInfo() {
		request = std::make_unique<NetworkInfoRequest>();
		response = std::make_unique<NetworkInfoResponse>();
		if (!request || !response) {
			LogError("Can't construct NetworkInfo.\n");
		}
	}

	virtual ~NetworkInfo() {}
};



template<typename T> std::unique_ptr<Data> create_vgmpad_data()
{
	std::unique_ptr<Data> vgmpad_data = T::Create();

	return std::move(vgmpad_data);
}



inline auto set_pkt = [](auto &pkt, auto val) -> void {
	auto sz = sizeof(val);
	for (auto i = 0; i < sz; i++) {
		auto e = (val >> (8 * (sz - i - 1))) & 0xFF;
		pkt.push_back(e);
	}
};

inline auto get_pkt = [](auto &itr, auto &val, auto &itr_end) -> void {
	auto sz = sizeof(val);
	val = 0;
	for (auto i = 0; (i < sz) && (itr < itr_end); i++) {
		val <<= 8;
		val |= *itr;
		itr++;
	}
};

class VISCA : public IData
{
protected:
	static constexpr uint8_t END_MARK = 0xFF;
	static constexpr uint8_t On = 0x02;
	static constexpr uint8_t Off = 0x03;

public:
	VISCA() {}
	virtual ~VISCA() {}
};



class VISCA_Command : public VISCA
{
protected:
	const packet_type head = { 0x81, 0x01 };

public:
	VISCA_Command() {}
	virtual ~VISCA_Command() {}
};



class VISCA_Response : public VISCA
{
public:
	enum class em_Type
	{
		ACK,
		Completion,
		Error,

		Invalid,
	};

	enum class em_ErrorCode
	{
		Message_length_eror = 0x01,
		Syntax_Error = 0x02,
		Command_buffer_full = 0x03,
		Command_canceled = 0x04,
		No_socket = 0x05,
		Command_not_executable = 0x41,
	};

private:
	em_Type type = em_Type::Invalid;
	int sock_num = -1;
	packet_type data;

protected:
	static constexpr uint8_t START_MARK = 0x90;
	static constexpr uint8_t ACK = 0x40;
	static constexpr uint8_t Completion = 0x50;
	static constexpr uint8_t Error = 0x60;

	static constexpr uint8_t MASK_TYPE = 0xF0;
	static constexpr uint8_t MASK_SOCKET = 0x0F;

public:
	packet_type pack() override
	{
		return {};	// TODO: implenent.
	}

	void unpack(const packet_type &pkt) override
	{
		type = em_Type::Invalid;
		sock_num = -1;
		data.clear();

		if (pkt.size() < 3 || pkt.back() != VISCA::END_MARK) return;

		auto st_mark = pkt[0];
		if (st_mark != START_MARK) return;

		auto t = pkt[1] & MASK_TYPE;
		switch(t) {
		case ACK:
			type = em_Type::ACK;
			break;

		case Completion:
			type = em_Type::Completion;
			break;

		case Error:
			type = em_Type::Error;
			break;

		default:
			;
		}

		sock_num = pkt[1] & MASK_SOCKET;

		data.assign(pkt.begin() + 2, pkt.end() - 1);
	}

	em_Type get_type() const { return type; }
	int get_socket_number() const { return sock_num; }
	const packet_type &get_data() const { return data; }
	bool is_ack() const { return (type == em_Type::ACK) && (sock_num >= 0) && data.empty(); }
	bool is_comp_command() const { return (type == em_Type::Completion) && (sock_num >= 0) && data.empty(); }
	bool is_comp_inquiry() const { return (type == em_Type::Completion) && (sock_num >= 0) && (data.size() > 0); }

	VISCA_Response() {}
	virtual ~VISCA_Response() {}
};



class VISCA_Tally_Command : public VISCA_Command
{
public:
	enum class em_COLOR
	{
		RED,
		GREEN,
	};

private:
	static constexpr uint8_t Category_Code = 0x7E;
	const packet_type RED = { 0x01, 0x0A };
	const packet_type GREEN = { 0x04, 0x1A };

	em_COLOR color = em_COLOR::GREEN;
	bool is_on = false;

public:
	packet_type pack() override
	{
		packet_type pkt;

		pkt = head;
		pkt.push_back(Category_Code);
		if (color == em_COLOR::RED) {
			pkt.insert(pkt.end(), RED.cbegin(), RED.cend());
		} else if (color == em_COLOR::GREEN) {
			pkt.insert(pkt.end(), GREEN.cbegin(), GREEN.cend());
		} else {
			return {};
		}
		pkt.push_back(0);
		pkt.push_back(is_on ? VISCA::On : VISCA::Off);
		pkt.push_back(VISCA::END_MARK);

		return pkt;
	}

	void unpack(const packet_type &pkt) override
	{
		;	// TODO: implement.
	}

	void set_tally_color(em_COLOR val) { color = val; }
	void set_tally_on(bool val) { is_on = val; }

	VISCA_Tally_Command() {}

	virtual ~VISCA_Tally_Command() {}
};



class VISCA_PanTilt_Command : public VISCA_Command
{
public:
	enum class em_UpDown
	{
		UP = 0x01,
		DOWN = 0x02,
		STOP = 0x03,
	};

	enum class em_LeftRight
	{
		LEFT = 0x01,
		RIGHT = 0x02,
		STOP = 0x03,
	};

	static constexpr uint8_t SPEED_MIN = 1;
	static constexpr uint8_t SPEED_MAX = 50;

private:
	static constexpr uint8_t Category_Code = 0x06;

	static constexpr uint8_t PanTiltMove_Code = 0x01;
	static constexpr uint8_t PanTiltHome_Code = 0x04;
	static constexpr uint8_t PanTiltReset_Code = 0x05;

	uint8_t pan_speed = SPEED_MIN;
	uint8_t tilt_speed = SPEED_MIN;
	em_UpDown up_down = em_UpDown::STOP;
	em_LeftRight left_right = em_LeftRight::STOP;
	std::vector<uint8_t> cmd_data;

public:
	packet_type pack() override
	{
		packet_type pkt;

		pkt = head;
		pkt.push_back(Category_Code);
		pkt.insert(pkt.end(), cmd_data.begin(), cmd_data.end());
		pkt.push_back(VISCA::END_MARK);

		return pkt;
	}

	void unpack(const packet_type &pkt) override
	{
		;	// TODO: implement.
	}

	void set_pan_tilt(uint8_t pan, uint8_t tilt, em_LeftRight lr, em_UpDown ud)
	{
		pan_speed = std::clamp(pan, SPEED_MIN, SPEED_MAX);
		tilt_speed = std::clamp(tilt, SPEED_MIN, SPEED_MAX);
		up_down = ud;
		left_right = lr;

		cmd_data.clear();
		cmd_data.push_back(PanTiltMove_Code);
		cmd_data.push_back(pan_speed);
		cmd_data.push_back(tilt_speed);
		cmd_data.push_back(static_cast<uint8_t>(left_right));
		cmd_data.push_back(static_cast<uint8_t>(up_down));
	}
	void set_up    (uint8_t val) { set_pan_tilt(SPEED_MIN, val, em_LeftRight::STOP, em_UpDown::UP); }
	void set_down  (uint8_t val) { set_pan_tilt(SPEED_MIN, val, em_LeftRight::STOP, em_UpDown::DOWN); }
	void set_left  (uint8_t val) { set_pan_tilt(val, SPEED_MIN, em_LeftRight::LEFT, em_UpDown::STOP); }
	void set_right (uint8_t val) { set_pan_tilt(val, SPEED_MIN, em_LeftRight::RIGHT, em_UpDown::STOP); }
	void set_up_left   (uint8_t pan, uint8_t tilt) { set_pan_tilt(pan, tilt, em_LeftRight::LEFT, em_UpDown::UP); }
	void set_up_right  (uint8_t pan, uint8_t tilt) { set_pan_tilt(pan, tilt, em_LeftRight::RIGHT, em_UpDown::UP); }
	void set_down_left (uint8_t pan, uint8_t tilt) { set_pan_tilt(pan, tilt, em_LeftRight::LEFT, em_UpDown::DOWN); }
	void set_down_right(uint8_t pan, uint8_t tilt) { set_pan_tilt(pan, tilt, em_LeftRight::RIGHT, em_UpDown::DOWN); }
	void set_stop() { set_pan_tilt(SPEED_MIN, SPEED_MIN, em_LeftRight::STOP, em_UpDown::STOP); }

	void set_home() {
		cmd_data.clear();
		cmd_data.push_back(PanTiltHome_Code);
	}

	void set_reset() {
		cmd_data.clear();
		cmd_data.push_back(PanTiltReset_Code);
	}

	VISCA_PanTilt_Command() {}

	virtual ~VISCA_PanTilt_Command() {}
};



class VISCA_Zoom_Command : public VISCA_Command
{
public:
	enum class em_TeleWide
	{
		TELE = 0x02,
		WIDE = 0x03,
		STOP = 0x00,
	};

	static constexpr auto SPEED_MIN = 0;
	static constexpr auto SPEED_MAX = 7;
	static constexpr auto SPEED_HIGHRESO_MIN = 0;
	static constexpr auto SPEED_HIGHRESO_MAX = 0x7FFE;

private:
	const uint8_t Category_Code = 0x04;
	const uint8_t Ext_Category_Code = 0x7E;

	static constexpr uint8_t Zoom_Code = 0x07;
	static constexpr uint8_t ZoomHighReso_Code = 0x17;

	int zoom_speed = SPEED_MIN;
	int zoom_highreso_speed = SPEED_HIGHRESO_MIN;
	em_TeleWide tele_wide = em_TeleWide::STOP;
	std::vector<uint8_t> cmd_data;

	bool is_highreso = false;

public:
	packet_type pack() override
	{
		packet_type pkt;

		pkt = head;
		if (is_highreso) pkt.push_back(Ext_Category_Code);
		pkt.push_back(Category_Code);
		pkt.insert(pkt.end(), cmd_data.begin(), cmd_data.end());
		pkt.push_back(VISCA::END_MARK);

		return pkt;
	}

	void unpack(const packet_type &pkt) override
	{
		;	// TODO: implement.
	}

	void set_zoom(int zoom, em_TeleWide tw)
	{
		zoom_speed = std::clamp(zoom, SPEED_MIN, SPEED_MAX);
		tele_wide = tw;

		cmd_data.clear();
		cmd_data.push_back(Zoom_Code);
		uint8_t val = (static_cast<uint8_t>(tele_wide) << 4) | (zoom_speed & 0xF);
		cmd_data.push_back(val);

		disable_highreso();
	}
	void set_tele(int val) { set_zoom(val, em_TeleWide::TELE); }
	void set_wide(int val) { set_zoom(val, em_TeleWide::WIDE); }
	void set_stop() { set_zoom(0, em_TeleWide::STOP); }

	void set_zoom_highreso(int zoom, em_TeleWide tw)
	{
		zoom_highreso_speed = std::clamp(zoom, SPEED_HIGHRESO_MIN, SPEED_HIGHRESO_MAX);
		tele_wide = tw;

		cmd_data.clear();
		cmd_data.push_back(ZoomHighReso_Code);
		cmd_data.push_back(static_cast<uint8_t>(tele_wide));
		cmd_data.push_back((zoom_highreso_speed >> 12) & 0x0F);
		cmd_data.push_back((zoom_highreso_speed >> 8) & 0x0F);
		cmd_data.push_back((zoom_highreso_speed >> 4) & 0x0F);
		cmd_data.push_back(zoom_highreso_speed & 0x0F);

		enable_highreso();
	}
	void set_tele_highreso(int val) { set_zoom_highreso(val, em_TeleWide::TELE); }
	void set_wide_highreso(int val) { set_zoom_highreso(val, em_TeleWide::WIDE); }
	void set_stop_highreso() { set_zoom_highreso(0, em_TeleWide::STOP); }

	void set_highreso(bool val) { is_highreso = val; }
	void enable_highreso() { set_highreso(true); }
	void disable_highreso() { set_highreso(false); }

	VISCA_Zoom_Command() {}

	virtual ~VISCA_Zoom_Command() {}
};



class VISCA_Focus_Command : public VISCA_Command
{
public:
	enum class em_NearFar
	{
		FAR = 0x02,
		NEAR = 0x03,
		STOP = 0x00,
	};

	enum class em_Mode
	{
		AUTO = 0x02,
		MANUAL = 0x03,
		TOGGLE = 0x10,
	};

	static constexpr auto SPEED_MIN = 0;
	static constexpr auto SPEED_MAX = 7;

private:
	const uint8_t Category_Code = 0x04;

	static constexpr uint8_t Focus_Code = 0x08;
	static constexpr uint8_t Mode_Code = 0x38;

	int focus_speed = SPEED_MIN;
	em_NearFar near_far = em_NearFar::STOP;
	std::vector<uint8_t> cmd_data;

	em_Mode mode = em_Mode::MANUAL;

public:
	packet_type pack() override
	{
		packet_type pkt;

		pkt = head;
		pkt.push_back(Category_Code);
		pkt.insert(pkt.end(), cmd_data.begin(), cmd_data.end());
		pkt.push_back(VISCA::END_MARK);

		return pkt;
	}

	void unpack(const packet_type &pkt) override
	{
		;	// TODO: implement.
	}

	void set_focus(int focus, em_NearFar nf)
	{
		focus_speed = std::clamp(focus, SPEED_MIN, SPEED_MAX);
		near_far = nf;

		cmd_data.clear();
		cmd_data.push_back(Focus_Code);
		uint8_t val = (static_cast<uint8_t>(near_far) << 4) | (focus_speed & 0xF);
		cmd_data.push_back(val);
	}
	void set_near(int val) { set_focus(val, em_NearFar::NEAR); }
	void set_far(int val) { set_focus(val, em_NearFar::FAR); }
	void set_stop(){ set_focus(0, em_NearFar::STOP); }

	void set_mode(em_Mode val)
	{
		mode = val;

		cmd_data.clear();
		cmd_data.push_back(Mode_Code);
		cmd_data.push_back(static_cast<uint8_t>(mode));
	}
	void set_mode_auto() { set_mode(em_Mode::AUTO); }
	void set_mode_manual() { set_mode(em_Mode::MANUAL); }
	void set_mode_toggle() { set_mode(em_Mode::TOGGLE); }

	VISCA_Focus_Command() {}

	virtual ~VISCA_Focus_Command() {}
};



class VISCA_IP : public IData
{
public:
	static constexpr uint16_t ID_COMMAND          = 0x01'00;
	static constexpr uint16_t ID_INQUIRY          = 0x01'10;
	static constexpr uint16_t ID_RESPONSE         = 0x01'11;
	static constexpr uint16_t ID_SETTING_COMMAND  = 0x01'20;
	static constexpr uint16_t ID_CONTROL          = 0x02'00;
	static constexpr uint16_t ID_CONTROL_RESPONSE = 0x02'01;

private:
	static uint32_t sequence_number;

	struct st_Header
	{
		uint16_t id = static_cast<uint16_t>(-1);
		uint16_t len = static_cast<uint16_t>(-1);
		uint32_t seq = static_cast<uint16_t>(-1);
	};

	st_Header header = {};
	packet_type payload = {};

protected:

public:
	static void reset_sequence_number() { sequence_number = 0; }
	static void count_up_sequence_number() { sequence_number++; }

	static bool is_error_invalid_sequence_number(const packet_type &pkt)
	{
		VISCA_IP visca_ip(pkt);
		bool is_error = (visca_ip.get_id() == ID_CONTROL)
			&& (visca_ip.get_payload() == packet_type{ 0x0F, 0x01 });

		return is_error;
	}
	bool is_error_invalid_sequence_number() const
	{
		bool is_error = (header.id == ID_CONTROL)
			&& (payload == packet_type{ 0x0F, 0x01 });

		return is_error;
	}

	static std::optional<VISCA_Response> get_msg_response(const packet_type &pkt)
	{
		VISCA_IP visca_ip(pkt);

		bool is_response = (visca_ip.get_id() == ID_RESPONSE);
		if (!is_response) return std::nullopt;

		auto payload = visca_ip.get_payload();
		VISCA_Response visca_res;
		visca_res.unpack(payload);

		return visca_res;
	}

	static bool is_ack(const packet_type &pkt)
	{
		auto visca_res = get_msg_response(pkt);
		if (!visca_res) return false;

		auto ret = visca_res->is_ack();

		return ret;
	}
	bool is_ack() const
	{
		if (header.id != ID_RESPONSE) return false;

		VISCA_Response visca_res;
		visca_res.unpack(payload);

		return visca_res.is_ack();
	}

	static bool is_comp_command(const packet_type &pkt)
	{
		auto visca_res = get_msg_response(pkt);
		if (!visca_res) return false;

		auto ret = visca_res->is_comp_command();

		return ret;
	}
	bool is_comp_command() const
	{
		if (header.id != ID_RESPONSE) return false;

		VISCA_Response visca_res;
		visca_res.unpack(payload);

		return visca_res.is_comp_command();
	}

	static bool is_comp_inquiry(const packet_type &pkt)
	{
		auto visca_res = get_msg_response(pkt);
		if (!visca_res) return false;

		auto ret = visca_res->is_comp_inquiry();

		return ret;
	}
	bool is_comp_inquiry() const
	{
		if (header.id != ID_RESPONSE) return false;

		VISCA_Response visca_res;
		visca_res.unpack(payload);

		return visca_res.is_comp_inquiry();
	}

	static int get_msg_socket_number(const packet_type &pkt)
	{
		auto visca_res = get_msg_response(pkt);
		if (!visca_res) return -1;

		auto ret = visca_res->get_socket_number();

		return ret;
	}

	static packet_type get_msg_data(const packet_type &pkt)
	{
		auto visca_res = get_msg_response(pkt);
		if (!visca_res) return {};

		auto ret = visca_res->get_data();

		return ret;
	}

	static packet_type make_packet_reset_sequence_number()
	{
		packet_type pkt{ 0x01 };

		VISCA_IP v_ip;
		v_ip.set_id(VISCA_IP::ID_CONTROL);
		v_ip.set_length(pkt.size());
		v_ip.set_payload(pkt);
		pkt = v_ip.pack();

		return pkt;
	}

	static bool is_ack_control_cmd(const packet_type &pkt)
	{
		VISCA_IP v_ip(pkt);

		bool is_ack_cc = (v_ip.get_id() == ID_CONTROL_RESPONSE)
			&& (v_ip.get_payload() == packet_type{ 0x01 });

		return is_ack_cc;
	}

	packet_type pack() override
	{
		packet_type pkt;

		set_pkt(pkt, header.id);
		set_pkt(pkt, header.len);
		header.seq = sequence_number;
		set_pkt(pkt, header.seq);

		pkt.insert(pkt.end(), payload.cbegin(), payload.cend());

		return pkt;
	}

	void unpack(const packet_type &pkt) override
	{
		auto itr = pkt.cbegin();
		auto itr_end = pkt.cend();

		get_pkt(itr, header.id, itr_end);
		get_pkt(itr, header.len, itr_end);
		get_pkt(itr, header.seq, itr_end);

		payload.assign(itr, pkt.cend());
	}

	void set_id(uint16_t val) { header.id = val; }
	void set_length(uint16_t val) { header.len = val; }
	void set_sequence_number(uint32_t val) { header.seq = val; }
	void set_payload(const packet_type &val) { payload = val; }

	void set_command(const packet_type &payload)
	{
		set_id(IpNetwork::VISCA_IP::ID_COMMAND);
		set_length(payload.size());
		set_payload(payload);
	}

	uint16_t get_id() const { return header.id; }
	uint16_t get_length() const { return header.len; }
	uint32_t get_sequence_number() const { return header.seq; }
	const packet_type &get_payload() const { return payload; }

	VISCA_IP() {}
	explicit VISCA_IP(const packet_type &pkt)
	{
		unpack(pkt);
	}

	virtual ~VISCA_IP() {}
};



class Connection
{
public:
	using queue_type = std::queue<packet_type>;
	static constexpr size_t QUEUE_SIZE_MAX = 1024;

	enum class em_Mode : int {
		SEND,	// target.
		RECEIVE,	// source.
		NONE,
	};

private:
	queue_type send_queue;
	queue_type receive_queue;
	std::mutex mtx_send_queue;
	std::mutex mtx_receive_queue;

	std::optional<packet_type> get_packet(queue_type &dataqueue, bool latest, bool keep_one)
	{
		std::optional<packet_type> pkt = std::nullopt;

		if (latest) {
			while (dataqueue.size() > 1) dataqueue.pop();
		}

		if (!dataqueue.empty()) {
			pkt = dataqueue.front();
			if (!keep_one || dataqueue.size() > 1) {
				dataqueue.pop();
			}
		}

		return pkt;
	}

	bool set_packet(const packet_type &pkt, queue_type &dataqueue, bool latest)
	{
		bool ret = true;

		if (latest) {
			auto is_full = [&]() -> bool {return (dataqueue.size() >= QUEUE_SIZE_MAX); };
			if (is_full()) {
				LogError("ERROR queue size is max in set_packet()\n");
				ret = false;
			}
			while (is_full()) dataqueue.pop();
		}

		if (dataqueue.size() < QUEUE_SIZE_MAX) {
			dataqueue.push(pkt);
		} else {
			LogError("ERROR queue size is max in set_packet()\n");
			ret = false;
		}

		return ret;
	}

protected:
	static constexpr size_t RECV_PACKET_SIZE_MAX = 10 * 1024;
	static constexpr auto RECV_RETRY_MAX = 10;	// TODO: toriaezu, x10 kaio-ken made.
	static constexpr auto EPORLL_WAIT_TIMEOUT = 1;

	StopWatch sw_timeout;

	em_Mode m_mode = em_Mode::NONE;
	std::string m_name;
	std::string m_service;
	bool is_caller = false;
	addrinfo *m_ai = nullptr;

	std::string m_mc_if;
	int m_mc_ttl = 1;
	std::unique_ptr<group_req> m_mc_gr = nullptr;

	bool is_broadcast = false;

	static void print_create_info(int ret, em_Mode mode, bool is_caller, const std::string &name, const std::string &service, const std::string &mc_if, int mc_ttl, bool is_broadcast)
	{
		LogInfo("======== Virtual Gamepad ========\n");

		if (!ret) {
			LogError("virtual gamepad -- can't opened\n");
		} else {
			std::string mode_str;
			if (mode == em_Mode::RECEIVE) {
				mode_str = "Receive";
			} else if (mode == em_Mode::SEND) {
				mode_str = "Send";
			} else {
				mode_str = "None";
			}

			LogSuccess("virtual gamepad -- opened(%s-%s) %s:%s --- [multicast] %s, %d --- [broadcast] %s\n",
				mode_str.c_str(), is_caller ? "caller" : "listener", name.c_str(), service.c_str(),
				mc_if.c_str(), mc_ttl,
				is_broadcast ? "true" : "false");
		}
	}

	static const char *get_direction_str(em_Mode mode)
	{
		return mode == em_Mode::RECEIVE ? "source" : "target";
	}

	std::optional<packet_type> get_send_packet(bool latest = false, bool keep_one = false)
	{
		std::lock_guard<std::mutex> lock(mtx_send_queue);
		return get_packet(send_queue, latest, keep_one);
	}
	std::optional<packet_type> get_receive_packet(bool latest = false, bool keep_one = false)
	{
		std::lock_guard<std::mutex> lock(mtx_receive_queue);
		return get_packet(receive_queue, latest, keep_one);
	}

	bool set_send_packet(const packet_type &pkt, bool latest = false)
	{
		std::lock_guard<std::mutex> lock(mtx_send_queue);
		return set_packet(pkt, send_queue, latest);
	}
	bool set_receive_packet(const packet_type &pkt, bool latest = false)
	{
		std::lock_guard<std::mutex> lock(mtx_receive_queue);
		return set_packet(pkt, receive_queue, latest);
	}

public:
	size_t get_send_queue_size() const { return send_queue.size(); }
	size_t get_receive_queue_size() const { return receive_queue.size(); }
	bool is_empty_send_queue() const { return send_queue.empty(); }
	bool is_empty_receive_queue() const { return receive_queue.empty(); }

	template<typename T>
	static std::unique_ptr<T> Create(const std::string &name, const std::string &service, em_Mode mode, bool is_caller, const std::string &mc_if, int mc_ttl, bool is_broadcast)
	{
		auto vgmpad = std::make_unique<T>();
		if (!vgmpad) return nullptr;

		auto ret = vgmpad->open(name, service, mode, is_caller, mc_if, mc_ttl, is_broadcast);
		if (!ret) {
			vgmpad.reset();
		}

		print_create_info(ret, mode, is_caller, name, service, mc_if, mc_ttl, is_broadcast);

		return std::move(vgmpad);
	}

	virtual bool open(const std::string &name, const std::string &service, em_Mode mode, bool is_caller, const std::string &mc_if, int mc_ttl, bool is_broadcast) = 0;
	virtual bool close() = 0;
	virtual bool poll(int64_t time_out = 33) = 0;

	virtual bool send(const packet_type &pkt)
	{
		return set_send_packet(pkt);
	}

	virtual std::optional<packet_type> receive()
	{
		return get_receive_packet();
	}

	Connection() {}

	virtual ~Connection() {}
};



class UDP_base : public Connection
{
private:

protected:
	int m_sock = -1;

	enum class em_SOCKSTATUS : int {
		INIT = 1,
		OPENED,
		LISTENING,
		CONNECTING,
		CONNECTED,
		BROKEN,
		CLOSING,
		CLOSED,
		NONEXIST
	};

	em_SOCKSTATUS m_sock_status = em_SOCKSTATUS::NONEXIST;

	em_SOCKSTATUS get_sock_status() const { return m_sock_status; }

#ifdef USE_EPOLL
	// epoll.
	static constexpr auto EPOLL_EVENT_MAX = 10;
	int m_pollid = -1;
	epoll_event m_sys_event[EPOLL_EVENT_MAX];
	int m_sys_event_len = std::size(m_sys_event);

	void print_epoll_events(uint32_t events)
	{
		if (events & EPOLLIN)        { LogError("event is EPOLLIN.\n"); }
		if (events & EPOLLPRI)       { LogError("event is EPOLLPRI.\n"); }
		if (events & EPOLLOUT)       { LogError("event is EPOLLOUT.\n"); }
		if (events & EPOLLRDNORM)    { LogError("event is EPOLLRDNORM.\n"); }
		if (events & EPOLLRDBAND)    { LogError("event is EPOLLRDBAND.\n"); }
		if (events & EPOLLWRNORM)    { LogError("event is EPOLLWRNORM.\n"); }
		if (events & EPOLLWRBAND)    { LogError("event is EPOLLWRBAND.\n"); }
		if (events & EPOLLMSG)       { LogError("event is EPOLLMSG.\n"); }
		if (events & EPOLLERR)       { LogError("event is EPOLLERR.\n"); }
		if (events & EPOLLHUP)       { LogError("event is EPOLLHUP.\n"); }
		if (events & EPOLLRDHUP)     { LogError("event is EPOLLRDHUP.\n"); }
		if (events & EPOLLEXCLUSIVE) { LogError("event is EPOLLEXCLUSIVE.\n"); }
		if (events & EPOLLWAKEUP)    { LogError("event is EPOLLWAKEUP.\n"); }
		if (events & EPOLLONESHOT)   { LogError("event is EPOLLONESHOT.\n"); }
		if (events & EPOLLET)        { LogError("event is EPOLLET.\n"); }
	}
#endif

public:
	virtual bool open(const std::string &name, const std::string &service, em_Mode mode, bool is_caller, bool connectionless, bool is_udp, const std::string &mc_if, int mc_ttl, bool is_broadcast)
	{
		m_mode = mode;
		m_name = name;
		m_service = service;

		this->is_caller = is_caller;
		m_mc_if = mc_if;
		m_mc_ttl = mc_ttl;
		addrinfo *ai_mc = nullptr;

		this->is_broadcast = is_broadcast;

		// is_caller = (name != "" && name != "0.0.0.0") ? true : false;

		auto open_error_return = [&]() -> bool {
			close();
			if (ai_mc) freeaddrinfo(ai_mc);
			return false;
		};

		// get addrinfo.
		{
			addrinfo fo = is_udp
				? addrinfo{	// UDP.
					AI_PASSIVE,
					AF_UNSPEC,
					SOCK_DGRAM, IPPROTO_UDP,
					0, 0,
					NULL, NULL
				}
				: addrinfo{	// TCP.
					AI_PASSIVE,
					AF_UNSPEC,
					SOCK_STREAM, IPPROTO_TCP,
					0, 0,
					NULL, NULL
				};
			const char *n = (name.empty() || name == "") ? nullptr : name.c_str();
			const char *s = (service.empty() || service == "") ? nullptr : service.c_str();
			int erc = getaddrinfo(n, s, &fo, &m_ai);
			if (erc != 0)
			{
				LogError("ERROR!! getaddrinfo(errno=%d): name=%s, service=%s.\n", erc, name.c_str(), service.c_str());
				return false;
			}
		}

		bool is_multicast = check_multicast(m_ai);

		m_sock = socket(m_ai->ai_family, m_ai->ai_socktype, 0);
		if ( m_sock == -1 ) {
			LogError("ERROR!! create socket\n");
			return open_error_return();
		}
		m_sock_status = em_SOCKSTATUS::OPENED;

		// pre config.
		int yes = 1;
		setsockopt(m_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof yes);

		if (connectionless) {
			// caller.
			if (is_caller) {
				if (is_multicast) {
					// multicast interface & TTL.
					set_multicast_caller(m_sock, mc_if, mc_ttl, m_ai->ai_family);
				}

				// broadcast.
				if (is_broadcast) {
					int yes = 1;
					setsockopt(m_sock, SOL_SOCKET, SO_BROADCAST, (const char*)&yes, sizeof(yes));
				}

			// listener.
			} else {
				if (!is_multicast) {
					if (bind(m_sock, m_ai->ai_addr, m_ai->ai_addrlen) != 0) {
						LogError("ERROR!! bind: errno = %d\n", errno);
						return open_error_return();
					}

				} else {
					auto family = m_ai->ai_family;
					addrinfo fo = addrinfo{
						AI_PASSIVE,
						family,
						SOCK_DGRAM, IPPROTO_UDP,
						0, 0,
						NULL, NULL
					};
					const char *s = (service.empty() || service == "") ? nullptr : service.c_str();
					int erc = getaddrinfo(nullptr, s, &fo, &ai_mc);	// always 0.0.0.0(ipv4) or ::(ipv6).
					if (erc != 0)
					{
						LogError("ERROR!! [Multicast] getaddrinfo(errno=%d): service=%s.\n", erc, service.c_str());
						return open_error_return();
					}

					if (bind(m_sock, ai_mc->ai_addr, ai_mc->ai_addrlen) != 0) {
						LogError("ERROR!! [Multicast] bind: errno = %d\n", errno);
						return open_error_return();
					}
					if (ai_mc) freeaddrinfo(ai_mc);

					// multicast group.
					if (!set_multicast_group(m_sock, mc_if, m_ai, m_mc_gr)) {
						return open_error_return();
					}
				}
			}
			m_sock_status = em_SOCKSTATUS::CONNECTING;

		} else {
			// caller.
			if (is_caller) {
				set_fio_blocking_mode(m_sock);	// TODO: non-block mode.
				auto ret = connect(m_sock, m_ai->ai_addr, m_ai->ai_addrlen);		// TODO: non-block mode.
				set_fio_non_blocking_mode(m_sock);		// TODO: non-block mode.
				if(ret != 0) {
					LogError("ERROR!! connect: errno = %d\n", errno);
					return open_error_return();
				}
				m_sock_status = em_SOCKSTATUS::CONNECTED;

			// listener.
			} else {
				if (bind(m_sock, m_ai->ai_addr, m_ai->ai_addrlen) != 0) {
					LogError("ERROR!! bind: errno = %d\n", errno);
					return open_error_return();
				}
				m_sock_status = em_SOCKSTATUS::CONNECTING;

				LogVerbose(" listen...\n");

				if (listen(m_sock, 1) == -1) {
					LogError("ERROR!! listen: errno = %d\n", errno);
					return open_error_return();
				}
				m_sock_status = em_SOCKSTATUS::LISTENING;
			}
		}

		if (!set_fio_non_blocking_mode(m_sock))
		{
			LogError("ERROR!! fcntl O_NONBLOCK\n");
			return open_error_return();
		}

#ifdef USE_EPOLL
		epoll_event ev;
		if (mode == em_Mode::SEND) {
			ev.events = EPOLLIN | EPOLLOUT | EPOLLRDHUP;
		} else if (mode == em_Mode::RECEIVE) {
			ev.events = EPOLLIN | EPOLLOUT | EPOLLRDHUP;
		}
		ev.data.fd = m_sock;
		int ret = epoll_ctl(m_pollid, EPOLL_CTL_ADD, m_sock, &ev);
		if (ret == -1) {
			LogError("ERROR!! epoll_ctl: errno = %d\n", errno);
			return open_error_return();
		}
#endif

		const auto str_direction = get_direction_str(m_mode);
		LogInfo("SUCCESS!! open virtual gamepad(%s).\n", str_direction);

		return true;
	}

	virtual bool open(const std::string &name, const std::string &service, em_Mode mode, bool is_caller, const std::string &mc_if, int mc_ttl, bool is_broadcast) override
	{
		return open(name, service, mode, is_caller, true, true, mc_if, mc_ttl, is_broadcast);
	}

	virtual bool close() override
	{
		m_sock_status = em_SOCKSTATUS::CLOSING;

		bool ret = false;

		if (m_sock >= 0 && m_ai && m_mc_gr) {
			multicast_leave_group(m_sock, *m_mc_gr, m_ai->ai_family);
			m_mc_gr.reset();
		}

		if (m_ai) {
			freeaddrinfo(m_ai);
			m_ai = nullptr;
		}

		if (m_sock >= 0) {
#ifdef _WIN32
			int r = ::closesocket(m_sock);
#else
			int r = ::close(m_sock);
#endif
			m_sock = -1;
			if (r >= 0) ret = true;
		}

		m_sock_status = em_SOCKSTATUS::CLOSED;

		return ret;
	}

	UDP_base()
	{
#ifdef USE_EPOLL
		m_pollid = epoll_create(1);
		if (m_pollid == -1) {
			LogError("ERROR!! epoll_create\n");
			exit(EXIT_FAILURE);
		}
		LogInfo("m_pollid = %d\n", m_pollid);
#endif
		m_sock_status = em_SOCKSTATUS::INIT;
	}

	virtual ~UDP_base()
	{
		if (!close()) {
			LogError("ERROR!! IpNetwork close.\n");
		}

#ifdef USE_EPOLL
		if (m_pollid != -1) {
			::close(m_pollid);
		}
#endif
	}
};



class UDP : public UDP_base
{
private:
	static constexpr auto WAIT_FOR_RECONNECT = 500;	// [msec].

protected:

public:
	static std::unique_ptr<UDP> Create(const std::string &name, const std::string &service, em_Mode mode, bool is_caller, const std::string &mc_if, int mc_ttl, bool is_broadcast)
	{
		return Connection::Create<UDP>(name, service, mode, is_caller, mc_if, mc_ttl, is_broadcast);
	}

	bool poll(int64_t time_out = 33) override
	{
		if (m_sock == -1)
		{
			if (!open(m_name, m_service, m_mode, is_caller, m_mc_if, m_mc_ttl, is_broadcast)) return false;
		}

		const auto str_direction = get_direction_str(m_mode);

#ifdef USE_EPOLL
		int num = epoll_wait(m_pollid, m_sys_event, m_sys_event_len, time_out);
		if (num < 0) {
			LogError("ERROR!! epoll_wait\n");
			return false;
		} else if (num == 0) {
			// LogError("TIME OUT!! epoll_wait\n");
			return false;
		}

		for (auto i = 0; i < num; i++)
#endif
		{
#ifdef USE_EPOLL
			if (m_sys_event[i].data.fd != m_sock) {
				LogInfo("INFO : event is not applicable.");
				continue;
			}
#endif

			if (m_mode != em_Mode::RECEIVE && m_mode != em_Mode::SEND) return false;

#ifdef USE_EPOLL
			if ((m_sys_event[i].events & EPOLLOUT) != 0)
#endif
			{
				while (!is_empty_send_queue())
				{
					auto pkt = get_send_packet().value();

					int stat = sendto(m_sock, (const char *)pkt.data(), pkt.size(), 0, m_ai->ai_addr, m_ai->ai_addrlen);
					if (stat < 1)
					{
						LogError("ERROR!! send UDP packet.\n");
						close();	// to reconnect.
						std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_FOR_RECONNECT));
						return false;
					}
					else
					{
#if 0	// for debug. check sending timing of packet.
						static StopWatch sw;
						sw.stop();
						auto dt = sw.duration();
						sw.start();
						std::cout << std::fixed << std::setprecision(3) << dt << std::defaultfloat << " : ";
#endif
						std::cout << "[SEND packet] size = " << pkt.size() << std::endl;
						print_bytestream(pkt);
					}
				}
			}

#ifdef USE_EPOLL
			if ((m_sys_event[i].events & EPOLLIN) != 0)
#endif
			{
				if (m_sock >= 0)
				{
					auto retry_count = 0;
					while (retry_count++ < RECV_RETRY_MAX)
					{
						packet_type pkt(RECV_PACKET_SIZE_MAX);
						sockaddr_storage from_addr;
						socklen_t sin_size = sizeof(from_addr);
						const int stat = recvfrom(m_sock, (char *)pkt.data(), pkt.size(), 0, (sockaddr *)&from_addr, &sin_size);
						if (stat < 0)
						{
							// LogInfo("Empty packets\n");

							if (errno == EAGAIN	// maybe, no packet.
#ifdef _MSC_VER
								|| (WSAGetLastError() == WSAEWOULDBLOCK && !is_empty_receive_queue())	// [Windows] maybe, no packet.
#endif
							) break;

							return false;
						}
						else if (stat == 0)
						{
							break;
						}
						if (stat < pkt.size()) pkt.resize(stat);
						set_receive_packet(pkt);
					}
					// std::cout << "m_sock = " << m_sock << std::endl;

				} else {
					// LogInfo("Empty packets\n");
					return false;
				}

				// LogDebug("dataqueue size = %ld\n", dataqueue.size());
				if (is_empty_receive_queue()) return false;
			}

#ifdef USE_EPOLL
			if ((m_sys_event[i].events & (EPOLLIN | EPOLLOUT)) == 0) {
				LogError("ERROR!! -------- %s\n", str_direction);
				print_epoll_events(m_sys_event[i].events);
				return false;
			}
#endif
		}

		return true;
	}

	UDP() {}

	virtual ~UDP() {}
};



class VISCA_Com
{
private:
	static constexpr auto PORT_NUMBER = "52381";

	std::unique_ptr<Connection> sender;
	std::unique_ptr<Connection> receiver;

public:
	static std::unique_ptr<VISCA_Com> Create(const std::string &name)
	{
		auto v_com = std::make_unique<VISCA_Com>();
		if (!v_com) return nullptr;

		v_com->sender = UDP::Create(name, PORT_NUMBER, Connection::em_Mode::SEND, true, "", 0, false);
		v_com->receiver = UDP::Create("0.0.0.0", PORT_NUMBER, Connection::em_Mode::RECEIVE, false, "", 0, false);
		if (!v_com->sender || !v_com->receiver) return nullptr;

		return std::move(v_com);
	}

	bool open(const std::string &name)
	{
		auto flg_s = sender->open(name, PORT_NUMBER, Connection::em_Mode::SEND, true, "", 0, false);
		auto flg_r = receiver->open("0.0.0.0", PORT_NUMBER, Connection::em_Mode::RECEIVE, false, "", 0, false);

		return flg_s & flg_r;
	}

	bool close()
	{
		auto flg_s = sender->close();
		auto flg_r = receiver->close();

		return flg_s & flg_r;
	}

	// Reset for invalid sequence number.
	bool send_reset_sequence_number(int timeout_msec = 1000, int retry_count = 10)
	{
		auto reset_pkt = VISCA_IP::make_packet_reset_sequence_number();
		sender->send(reset_pkt);
		sender->poll();

		// wait ACK(Control Command).
		{
			bool is_recv_loop = true;
			for (auto i = 0; i < retry_count && is_recv_loop; i++) {
				while (receiver->poll(timeout_msec)) {
					if (receiver->is_empty_receive_queue()) break;

					while (auto pkt = receiver->receive()) {
						if (VISCA_IP::is_ack_control_cmd(*pkt)) {
							while (receiver->receive()) {}	// clear receive buffer.
							is_recv_loop = false;
							break;
						}
					}

					if (!is_recv_loop) break;
				}

				if (is_recv_loop) std::this_thread::sleep_for(std::chrono::milliseconds(33));
			}
		}

		return true;
	}

	bool send(VISCA_IP &visca_ip, int timeout_msec = 1000, int retry_count = 10)
	{
		// Send packet.
		auto pkt = visca_ip.pack();
		sender->send(pkt);
		sender->poll();

		// Receive Response.
		bool is_recv_loop = true;
		for (auto i = 0; i < retry_count && is_recv_loop; i++) {
			while (receiver->poll(timeout_msec)) {
				if (receiver->is_empty_receive_queue()) break;

				while (auto pkt = receiver->receive()) {
					VISCA_IP visca_ip(*pkt);

					// Reset for invalid sequence number.
					if (visca_ip.is_error_invalid_sequence_number()) {
						send_reset_sequence_number();
					}

					if (visca_ip.is_ack()) {
						std::cout << "Receive ACK." << std::endl;
					}

					if (visca_ip.is_comp_command()) {
						std::cout << "Receive Completion(command)." << std::endl;
						while (receiver->receive()) {}	// clear receive buffer.
						is_recv_loop = false;
						break;
					}
				}

				if (!is_recv_loop) break;
			}

			if (is_recv_loop) std::this_thread::sleep_for(std::chrono::milliseconds(33));
		}

		visca_ip.count_up_sequence_number();

		return true;
	}

	std::optional<packet_type> receive()
	{
		return {};
	}

	VISCA_IP make_visca_ip(VISCA_Command &cmd)
	{
		auto payload = cmd.pack();

		VISCA_IP visca_ip;
		visca_ip.set_command(payload);

		return visca_ip;
	}

	bool send_visca_ip(VISCA_Command &cmd)
	{
		auto visca_ip = make_visca_ip(cmd);
		auto ret = send(visca_ip);

		return ret;
	}

	// Send Tally command.
	bool send_cmd_tally(VISCA_Tally_Command::em_COLOR color, bool is_on)
	{
		VISCA_Tally_Command cmd;
		cmd.set_tally_color(color);
		cmd.set_tally_on(is_on);

		return send_visca_ip(cmd);
	}

	// Send Pan Tilt command.
	bool send_cmd_pan_tilt(uint8_t pan, uint8_t tilt, VISCA_PanTilt_Command::em_LeftRight lr, VISCA_PanTilt_Command::em_UpDown ud)
	{
		VISCA_PanTilt_Command cmd;
		cmd.set_pan_tilt(pan, tilt, lr, ud);

		return send_visca_ip(cmd);
	}
	bool send_pt_up(uint8_t val)
	{
		VISCA_PanTilt_Command cmd;
		cmd.set_up(val);

		return send_visca_ip(cmd);
	}
	bool send_pt_down(uint8_t val)
	{
		VISCA_PanTilt_Command cmd;
		cmd.set_down(val);

		return send_visca_ip(cmd);
	}
	bool send_pt_left(uint8_t val)
	{
		VISCA_PanTilt_Command cmd;
		cmd.set_left(val);

		return send_visca_ip(cmd);
	}
	bool send_pt_right(uint8_t val)
	{
		VISCA_PanTilt_Command cmd;
		cmd.set_right(val);

		return send_visca_ip(cmd);
	}
	bool send_pt_up_left(uint8_t pan, uint8_t tilt)
	{
		VISCA_PanTilt_Command cmd;
		cmd.set_up_left(pan, tilt);

		return send_visca_ip(cmd);
	}
	bool send_pt_up_right(uint8_t pan, uint8_t tilt)
	{
		VISCA_PanTilt_Command cmd;
		cmd.set_up_right(pan, tilt);

		return send_visca_ip(cmd);
	}
	bool send_pt_down_left(uint8_t pan, uint8_t tilt)
	{
		VISCA_PanTilt_Command cmd;
		cmd.set_down_left(pan, tilt);

		return send_visca_ip(cmd);
	}
	bool send_pt_down_right(uint8_t pan, uint8_t tilt)
	{
		VISCA_PanTilt_Command cmd;
		cmd.set_down_right(pan, tilt);

		return send_visca_ip(cmd);
	}
	bool send_cmd_pt_stop()
	{
		VISCA_PanTilt_Command cmd;
		cmd.set_stop();

		return send_visca_ip(cmd);
	}

	bool send_cmd_pt_home()
	{
		VISCA_PanTilt_Command cmd;
		cmd.set_home();

		return send_visca_ip(cmd);
	}

	bool send_cmd_pt_reset()
	{
		VISCA_PanTilt_Command cmd;
		cmd.set_reset();

		return send_visca_ip(cmd);
	}

	// Send Zoom command.
	bool send_cmd_zoom(int val, VISCA_Zoom_Command::em_TeleWide tw, bool is_highreso = true)
	{
		VISCA_Zoom_Command cmd;
		if (is_highreso) {
			cmd.set_zoom_highreso(val, tw);
		} else {
			cmd.set_zoom(val >> 12, tw);
		}

		return send_visca_ip(cmd);
	}

	bool send_cmd_zm_tele(int val, bool is_highreso = true)
	{
		VISCA_Zoom_Command cmd;
		if (is_highreso) {
			cmd.set_tele_highreso(val);
		} else {
			cmd.set_tele(val >> 12);
		}

		return send_visca_ip(cmd);
	}

	bool send_cmd_zm_wide(int val, bool is_highreso = true)
	{
		VISCA_Zoom_Command cmd;
		if (is_highreso) {
			cmd.set_wide_highreso(val);
		} else {
			cmd.set_wide(val >> 12);
		}

		return send_visca_ip(cmd);
	}

	bool send_cmd_zm_stop(bool is_highreso = true)
	{
		VISCA_Zoom_Command cmd;
		if (is_highreso) {
			cmd.set_stop_highreso();
		} else {
			cmd.set_stop();
		}

		return send_visca_ip(cmd);
	}

	// Send Focus command.
	bool send_cmd_focus(int val, VISCA_Focus_Command::em_NearFar nf)
	{
		VISCA_Focus_Command cmd;
		cmd.set_focus(val, nf);

		return send_visca_ip(cmd);
	}

	bool send_cmd_focus_near(int val)
	{
		VISCA_Focus_Command cmd;
		cmd.set_near(val);

		return send_visca_ip(cmd);
	}

	bool send_cmd_focus_far(int val)
	{
		VISCA_Focus_Command cmd;
		cmd.set_far(val);

		return send_visca_ip(cmd);
	}

	bool send_cmd_focus_stop()
	{
		VISCA_Focus_Command cmd;
		cmd.set_stop();

		return send_visca_ip(cmd);
	}

	bool send_cmd_focus_mode(VISCA_Focus_Command::em_Mode mode)
	{
		VISCA_Focus_Command cmd;
		cmd.set_mode(mode);

		return send_visca_ip(cmd);
	}

	bool send_cmd_focus_auto()
	{
		VISCA_Focus_Command cmd;
		cmd.set_mode_auto();

		return send_visca_ip(cmd);
	}

	bool send_cmd_focus_manual()
	{
		VISCA_Focus_Command cmd;
		cmd.set_mode_manual();

		return send_visca_ip(cmd);
	}

	bool send_cmd_focus_toggle()
	{
		VISCA_Focus_Command cmd;
		cmd.set_mode_toggle();

		return send_visca_ip(cmd);
	}

	VISCA_Com() {}

	virtual ~VISCA_Com() {}

};

}	// namespace IpNetwork.



struct st_NetInfo
{
	std::string nickname;
	std::string ipadr;
	std::string mac;

public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        st_NetInfo,

		nickname,
		ipadr,
		mac
    )
};
inline bool operator<(const st_NetInfo& a, const st_NetInfo& b) noexcept {
	return std::tie(a.nickname, a.ipadr, a.mac) < std::tie(b.nickname, b.ipadr, b.mac);
}
extern void init_search_ipadr();
extern std::vector<st_NetInfo> search_ipadr();
