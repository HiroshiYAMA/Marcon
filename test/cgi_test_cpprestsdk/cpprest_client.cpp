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
#include <cpprest/http_client.h>

using namespace web;
using namespace web::http;
using namespace web::http::client;

#include "json.hpp"
using njson = nlohmann::json;

#include <thread>
#include <chrono>
#include <list>
/////////////////////////////////////////////////////////////////////
// StopWatch.
/////////////////////////////////////////////////////////////////////

class StopWatch
{
    using clock = std::chrono::steady_clock;
    static constexpr size_t queue_max = 100;
private:
    clock::time_point st, et;
    double ave;
    std::list<double> lap_list;

public:
    StopWatch()
    {
        start();
    }

    virtual ~StopWatch() {}

    clock::time_point start()
    {
        st = clock::now();
        return st;
    }

    clock::time_point stop()
    {
        et = clock::now();
        return et;
    }

    double duration()
    {
        auto dt = et - st;
        auto dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count();
        double dt_ms = dt_ns / 1'000.0f / 1'000.0f;

        lap_list.push_back(dt_ms);
        if (lap_list.size() > queue_max) lap_list.pop_front();

        return dt_ms;
    }

    double lap_ave()
    {
        ave = std::accumulate(lap_list.begin(), lap_list.end(), 0.0) / lap_list.size();
        return ave;
    }

    std::tuple<double, double> lap()
    {
        et = stop();
        auto dt_ms = duration();
        lap_ave();
        st = et;

        return { dt_ms, ave };
    }

};

pplx::task<void> Get(const std::string &host, int port, const std::string &msg)
{
	return pplx::create_task([=]()
	{
		uri_builder bld;
		bld.set_host(host);
		bld.set_port(port);
		bld.set_path(msg);

		credentials cred("admin", "Admin_1234");
		http_client_config cfg;
		cfg.set_credentials(cred);
		// std::cout
		// 	<< "chunksize: " << cfg.chunksize() << std::endl
		// 	<< "credentials: " << cfg.credentials().username() << std::endl
		// 	<< "guarantee_order: " << cfg.guarantee_order() << std::endl
		// 	<< "proxy: " << cfg.proxy().address().host() << std::endl
		// 	<< "timeout: " << cfg.timeout().count() << std::endl
		// 	<< "validate_certificates: " << cfg.validate_certificates() << std::endl
		// 	;
	
		http_client client(bld.to_uri(), cfg);

		auto uri = client.base_uri();
		std::cout
			<< "host: " << uri.host() << std::endl
			<< "port: " << uri.port() << std::endl
			<< "path: " << uri.path() << std::endl
			<< "query:" << uri.query() << std::endl
			<< "scheme: " << uri.scheme() << std::endl
			<< "to_string: " << uri.to_string() << std::endl
			<< "user_info: " << uri.user_info() << std::endl
			<< std::endl;

		http_request request(methods::GET);

		auto &headers = request.headers();
		std::stringstream ss;
		ss << "http://"
			<< host
			<< ":" << port
			<< "/";
		std::string referer_str = ss.str();
		headers.add("Referer", referer_str);

		return client.request(request);

	}).then([](http_response response)
	{
		auto status = response.status_code();
		if (status == status_codes::OK)
		{
			// レスポンスを文字列として取得後、標準出力する
			auto body = response.extract_string();
			std::cout << body.get().c_str() << std::endl;
		} else {
			std::cout << "[RESPONSE STATUS] : " << status << std::endl;
		}
	});
}

int main(int ac, char *av[])
{
	std::string server_adr = av[1];
	int server_port = std::stoi(av[2]);
	std::string cgi_message = av[3];

	StopWatch sw, sw2;

	// constexpr char *cmd_list[] = {
	// 	"/command/imaging.cgi?ExposureAutoShutterEnable=on",
	// 	"/command/imaging.cgi?ExposureShutterMode=speed&ExposureAutoShutterEnable=off&ExposureShutterSpeedEnable=on&ExposureECSEnable=off",
	// 	"/command/imaging.cgi?ExposureShutterMode=angle&ExposureAutoShutterEnable=off&ExposureECSEnable=off",
	// 	"/command/imaging.cgi?ExposureShutterMode=speed&ExposureAutoShutterEnable=off&ExposureShutterSpeedEnable=on&ExposureECSEnable=on",
	// 	"/command/imaging.cgi?ExposureShutterMode=speed&ExposureAutoShutterEnable=off&ExposureShutterSpeedEnable=off",
	// };

	try
	{
		pplx::task<void> a;
		for (auto i = 0; i < 1; i++) {
			// auto idx = i % std::size(cmd_list);
			sw.start();
			Get(server_adr, server_port, cgi_message).wait();
			// a = Get(server_adr, server_port, cgi_message);
			// a = Get(server_adr, server_port, cmd_list[idx]);
			auto [dt_ms, ave] = sw.lap();
			std::cout << "time: " << dt_ms << " , " << ave << std::endl;

			std::this_thread::sleep_for(std::chrono::milliseconds(33));
		}
		// for (auto i = 0; i < 10; i++) {
		// 	sw2.start();
		// 	a.wait();
		// 	auto [dt_ms2, ave2] = sw2.lap();
		// 	std::cout << "time(response): " << dt_ms2 << " , " << ave2 << std::endl;

		// 	std::this_thread::sleep_for(std::chrono::milliseconds(33));
		// }
	}
	catch (const std::exception &e)
	{
		std::cout << "Error " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// std::this_thread::sleep_for(std::chrono::milliseconds(1000));

	return EXIT_SUCCESS;
}
