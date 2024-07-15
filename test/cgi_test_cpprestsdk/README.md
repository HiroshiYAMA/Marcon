# ILME-FR7 と CGI で通信
CGI による通信するテストツール。

だがしかし、 Digest 認証をサポートしていないので使えない。

## 必要なもの
* C++ REST SDK

### cpp-httplib
C++ で HTTP な通信をするためのライブラリ。
* https://github.com/microsoft/cpprestsdk

```bash
sudo apt install libcpprest-dev
```

## ビルド & インストール
```bash
mkdir -p build
cd build
cmake ..
make

# build ディレクトリ内に実行ファイル cpprest_client が出来る。
```

## 実行
```bash
./cppret_client 43.30.217.166 80 '/command/inquiry.cgi?inq=system'
```
