# ILME-FR7 と CGI で通信
CGI による通信するテストツール。

## 必要なもの
* OpenSSL (ver.3)
* cpp-httplib

### OpenSSL ver.3
HTTP の Digest 認証のために必要(というより cpp-httplib のために必要)。
Ubuntu20.04 までの `apt` コマンドでインストールされる OpenSSL はきっと ver.1 系なので、自分でビルド & インストールする。

公式ページのとおりの手順で OK。
インストール先のディレクトリを設定するには下記を参照。
* https://github.com/openssl/openssl/blob/master/INSTALL.md#installing-to-a-different-location

### cpp-httplib
C++ で HTTP な通信をするためのライブラリ。
* https://github.com/yhirose/cpp-httplib

トップ階層の ***external*** ディレクトリに `git` の **submodule** として置いてある。

## ビルド & インストール
```bash
mkdir -p build
cd build
cmake ..
make

# build ディレクトリ内に実行ファイル client (と server) が出来る。
```

## 実行
```bash
./client 192.168.100.11 80 '/command/inquiry.cgi?inq=imaging'
```
