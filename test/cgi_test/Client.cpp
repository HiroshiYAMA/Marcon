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
