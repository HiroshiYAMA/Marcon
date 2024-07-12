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
#include <httplib.h>

#include <json.hpp>
using njson = nlohmann::json;

int main(int ac, char *av[])
{
	std::string server_adr = av[1];
	int server_port = std::stoi(av[2]);
	std::string cgi_message = av[3];

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

	auto res = cli.Get(cgi_message);

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

				try
				{
					auto j = njson::parse(res->body);
					std::cout << "JSON " << j << std::endl;
					auto s = j["system"];
					auto ss = s["LanguageInfo"];
					std::cout << "JSON: " << ss << std::endl;
					auto jj = j["/system/LanguageInfo"_json_pointer];
					std::cout << "JSON : " << jj << std::endl;
				}
				catch(const std::exception& e)
				{
					std::cerr << e.what() << '\n';
				}
			}
			break;
		default:
			;
		}
	}

	return EXIT_SUCCESS;
}
