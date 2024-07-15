# ILME-FR7 と CGI で通信
CGI による通信するテストツール。

## 必要なもの
* C++ Requests

### C++ Requests
C++ で HTTP な通信をするためのライブラリ。
* https://github.com/libcpr/cpr?tab=readme-ov-file

## ビルド & インストール
```bash
mkdir -p build
cd build
cmake ..
make

# build ディレクトリ内に実行ファイル cpr_client が出来る。
```

## 実行
```bash
./cpr_client 43.30.217.166 80 '/command/inquiry.cgi?inq=system'
```
