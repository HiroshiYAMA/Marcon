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

#include <json.hpp>
using njson = nlohmann::json;

pplx::task<void> Get(const std::string &host, int port, const std::string &msg)
{
	return pplx::create_task([=]()
	{
		uri_builder bld;
		bld.set_host(host);
		bld.set_port(port);
		bld.set_path(msg);

		credentials cred("user_name", "password");
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

		// auto uri = client.base_uri();
		// std::cout
		// 	<< "host: " << uri.host() << std::endl
		// 	<< "port: " << uri.port() << std::endl
		// 	<< "path: " << uri.path() << std::endl
		// 	<< "query:" << uri.query() << std::endl
		// 	<< "scheme: " << uri.scheme() << std::endl
		// 	<< "to_string: " << uri.to_string() << std::endl
		// 	<< "user_info: " << uri.user_info() << std::endl
		// 	<< std::endl;

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
		if (response.status_code() == status_codes::OK)
		{
			// レスポンスを文字列として取得後、標準出力する
			auto body = response.extract_string();
			std::cout << body.get().c_str() << std::endl;
		}
	});
}

int main(int ac, char *av[])
{
	std::string server_adr = av[1];
	int server_port = std::stoi(av[2]);
	std::string cgi_message = av[3];

	try
	{
		Get(server_adr, server_port, cgi_message).wait();
	}
	catch (const std::exception &e)
	{
		std::cout << "Error " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
