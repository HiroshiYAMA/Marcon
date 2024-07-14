# Marcon
ILME-FR7 用のリモコンツール(via IP ネットワーク)。

## ソース一式の取得(git clone)
```bash
git clone --recursive https://github.com/HiroshiYAMA/Marcon
又は、
git clone --recursive git@github.com:HiroshiYAMA/Marcon.git
```

もし、submodule を取得し忘れていたら、
```bash
git submodule update --init --recursive
```

## Requires
[C++ REST SDK](https://github.com/microsoft/cpprestsdk) をインストールする。
```bash
sudo apt install libcpprest-dev
```

## テストプログラムたち
test ディレクトリ以下に置いてある。
* [cgi_test](./test/cgi_test/)
* [srt_receive](./test/srt_receive/)
