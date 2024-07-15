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

#include <cpr/cpr.h>

int main(int a, char *av[])
{
	std::string server_adr = av[1];
	int server_port = std::stoi(av[2]);
	std::string cgi_message = av[3];

	StopWatch sw;

	std::stringstream ss;
	ss << "http://" << server_adr << ":" << server_port << cgi_message;
	std::string url_str = ss.str();
	auto url = cpr::Url{ url_str };

	auto auth = cpr::Authentication{ "user_name", "password", cpr::AuthMode::DIGEST };

	ss.clear();
	ss << "http://" << server_adr << ":" << server_port;
	std::string referer_str = ss.str();
	auto header = cpr::Header{
		{ "Referer", referer_str },
	};

	for (auto i = 0; i < 10; i++) {
		sw.start();
		cpr::Response r = cpr::Get(url, auth, header);
		auto [dt_ms, ave] = sw.lap();
		std::cout << "time: " << dt_ms << " , " << ave << std::endl;

		auto status = r.status_code;
		auto reason = r.reason;
		switch (status) {
		case cpr::status::HTTP_OK:
			std::cout << "(" << status << ")" << reason << std::endl;
			std::cout << "body : " << r.text << std::endl;
			break;
		case cpr::status::HTTP_UNAUTHORIZED:
			std::cout << "ERROR (" << status << ") : " << reason << std::endl;
			break;
		default:
			std::cout << "(" << status << ") : " << reason << std::endl;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(33));
	}

	return EXIT_SUCCESS;
}
