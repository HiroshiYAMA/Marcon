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
				*(sockaddr_in *)&sa = *((sockaddr_in *)ifa->ifa_ifu.ifu_broadaddr);
				break;
			case AF_INET6:
				*(sockaddr_in6 *)&sa = *((sockaddr_in6 *)ifa->ifa_ifu.ifu_broadaddr);
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

		if ((family == net_family) && (flags & flags_mask)) {
			sockaddr_storage sa = {};
			switch (family) {
			case AF_INET:
				*(sockaddr_in *)&sa = *((sockaddr_in *)ifa->ifa_ifu.ifu_broadaddr);
				break;
			case AF_INET6:
				*(sockaddr_in6 *)&sa = *((sockaddr_in6 *)ifa->ifa_ifu.ifu_broadaddr);
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
	virtual std::vector<uint8_t> pack() = 0;
	virtual void unpack(const std::vector<uint8_t> &pkt) = 0;
};



class Data : public IData
{
protected:
	static constexpr uint8_t START_MARK = 0x02;
	static constexpr uint8_t END_MARK = 0x03;
	static constexpr uint8_t DELIMITER_MARK = 0xFF;
	static constexpr uint8_t PARAMETER_MARK = ':';

	static bool check_data(const std::vector<uint8_t> &data)
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
	std::vector<uint8_t> pack() override
	{
		std::vector<uint8_t> pkt;

		pkt.push_back(START_MARK);

		const std::string str = std::string{PARAM} + (char)PARAMETER_MARK + VALUE;
		for (auto &e : str) {
			pkt.push_back(e);
		}
		pkt.push_back(DELIMITER_MARK);

		pkt.push_back(END_MARK);

		return pkt;
	}

	void unpack(const std::vector<uint8_t> &pkt) override
	{
		if (!check_data(pkt)) return;

		std::string str(pkt.begin() + 1, pkt.end() - 2);
		std::stringstream ss(str);
		std::vector<std::string> param_list;
		while (std::getline(ss, str, (char)0xFF)) {
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
	std::vector<uint8_t> pack() override
	{
		std::vector<uint8_t> pkt;

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
			for (auto &e : str) {
				pkt.push_back(e);
			}
			pkt.push_back(DELIMITER_MARK);
		}

		pkt.push_back(END_MARK);

		return pkt;
	}

	void unpack(const std::vector<uint8_t> &pkt) override
	{
		if (!check_data(pkt)) return;

		std::string str(pkt.begin() + 1, pkt.end() - 2);
		std::stringstream ss(str);
		std::vector<std::string> param_list;
		while (std::getline(ss, str, (char)0xFF)) {
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
	std::vector<uint8_t> pack() override
	{
		auto pkt = request->pack();

		return pkt;
	}

	void unpack(const std::vector<uint8_t> &pkt) override
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



class Connection
{
public:
	enum class em_Mode : int {
		SEND,	// target.
		RECEIVE,	// source.
		NONE,
	};

private:
	using packet_type = std::vector<uint8_t>;
	using queue_type = std::queue<packet_type>;
	static constexpr size_t QUEUE_SIZE_MAX = 1024;

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

	size_t get_send_queue_size() const { return send_queue.size(); }
	size_t get_receive_queue_size() const { return receive_queue.size(); }
	bool is_empty_send_queue() const { return send_queue.empty(); }
	bool is_empty_receive_queue() const { return receive_queue.empty(); }

public:
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

	virtual bool send(const packet_type &pkt, int64_t time_out = 33)
	{
		return set_send_packet(pkt);
	}

	virtual std::optional<packet_type> receive(int64_t time_out = 33)
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
		int num = epoll_wait(m_pollid, m_sys_event, m_sys_event_len, EPORLL_WAIT_TIMEOUT);
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
						std::vector<uint8_t> pkt(RECV_PACKET_SIZE_MAX);
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
