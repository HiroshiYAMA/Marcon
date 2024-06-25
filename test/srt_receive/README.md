# srt_receive
GStreamer の SRT プラグインを使って映像を受信 & 表示するテストツール。

## 必要なもの
* GStreamer
* SRT
* OpenCV

### Ubuntu20.04 以降
`apt` コマンドでインストールすれば OK。

```bash
sudo apt install libgstreamer1.0-0 gstreamer1.0-tools
sudo apt install gstreamer1.0-plugins-* gstreamer1.0-* libgstreamer-plugins-*-dev
sudo apt install libopencv-dev
```

### Ubuntu18.04

#### GStreamer SRT プラグイン
`apt` コマンドでインストールされる GStreamer のプラグインに SRT プラグインが含まれていない。
自分でビルドしてインストールする。

インストールは下記の Web ページを参考にすれば OK。
* https://zenn.dev/tetsu_koba/articles/ddc3902158b94a

ビルド & インストールしたライブラリファイルは、
***/usr/local/lib/gstreamer-1.0***
にあるので、
* libgstsrt.la
* libgstsrt.so

***/usr/lib/x86_64-linux-gnu/gstreamer-1.0***
にシンボリックリンク又はコピーすれば OK。

#### OpenCV
OpenCV のバージョンは ver.4 以降をインストールする。
apt コマンドで ver.3 がインストールされる場合は、自分でビルドしてインストールする。

```bash
# 最もシンプルな手順
# 必要に応じて ccmake で設定を変更する。

# OpenCV のソースファイルを展開後、
cd opencvほげほげ
mkdir -p build
cd build
ccmake -DCMAKE_BUILD_TYPE=Release ..
sudo make -j$(nproc) install
```

## ビルド & インストール
```bash
mkdir -p build
cd build
cmake ..
make install

# install ディレクトリ内に実行ファイル srt_receive が出来る。
```

## 実行
```bash
./srt_receive
```
