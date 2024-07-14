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

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"

#include "json.hpp"
using njson = nlohmann::json;

#include <thread>
#include <chrono>
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

inline auto print_sw_lap = [](StopWatch &sw, int &cnt, const std::string &str) -> void {
    auto [dt_ms, dt_ave] = sw.lap();
    if (cnt++ > 200) {
#ifdef USE_JETSON_UTILS
        // // cudaStreamSynchronize(NULL);
        // cudaDeviceSynchronize();
#endif
        std::cout << str << dt_ms << "(msec), " << dt_ave << "(msec)." << std::endl;
        cnt = 0;
    }
};

int main(int ac, char *av[])
{
	std::string server_adr = av[1];
	int server_port = std::stoi(av[2]);
	std::string cgi_message = av[3];

	StopWatch sw;

	// HTTP
	httplib::Client cli(server_adr, server_port);

	// HTTPS
	// httplib::Client cli("https://cpp-httplib-server.yhirose.repl.co");

	// Digest.
	cli.set_digest_auth("user_name", "password");

	// for Referer check.
	std::stringstream ss;
	ss << "http://"
		<< server_adr
		<< ":" << server_port
		<< "/";
	std::string referer_str = ss.str();
	cli.set_default_headers({
		{ "Referer", referer_str }
	});

	for (auto i = 0; i < 1; i++) {
		sw.start();
		auto res = cli.Get(cgi_message);
		auto [dt_ms, ave] = sw.lap();
		std::cout << "time: " << dt_ms << " , " << ave << std::endl;

		if (res) {
			auto st = res->status;
			auto msg = httplib::status_message(st);
			switch (st) {
				using SC = httplib::StatusCode;
			case SC::Unauthorized_401:
				std::cout << "ERROR (" << st << ") : " << msg << std::endl;
				break;
			case SC::OK_200:
				{
					std::cout << "(" << st << ")" << msg << std::endl;
					std::cout << "body : " << res->body << std::endl;

					// try
					// {
					// 	auto j = njson::parse(res->body);
					// 	std::cout << "JSON " << j << std::endl;
					// 	auto s = j["system"];
					// 	auto ss = s["LanguageInfo"];
					// 	std::cout << "JSON: " << ss << std::endl;
					// 	auto jj = j["/system/LanguageInfo"_json_pointer];
					// 	std::cout << "JSON : " << jj << std::endl;
					// }
					// catch(const std::exception& e)
					// {
					// 	std::cerr << e.what() << '\n';
					// }
				}
				break;
			default:
				;
			}
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(33));
	}

	return EXIT_SUCCESS;
}
