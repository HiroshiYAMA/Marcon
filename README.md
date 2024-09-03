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

## 必要なもの
* OpenSSL ver.3
  * 結構なシステムにインストールされているであろう OpenSSL ver.1 系の
ダイナミックリンクライブラリとの不意な競合を避けるには
スタティックリンクライブラリとしてビルドすると良い。

```bash
./Configure --prefix=/opt/openssl3 --openssldir=/usr/local/ssl3 zlib no-shared
make -j$(nproc)
make test
sudo make install
```

## ビルド & インストール

### ビルドオプションたち
CMakeLists.txtの設定項目は以下のとおり。
| オプション | 説明 |
| --- | --- |
| 変数 **CXX_STANDARD** | ビルドに使う C++ のバージョンを指定する。デフォルト **20**、必要に応じて **17** に変更する |
| 変数 **OPENSSL_ROOT_DIR** | OpenSSL v3 をインストールしているディレクトリを設定する |
| **target_compile_definitions(${target} PUBLIC GST_NV)** | H.264 のデコードに GStreamer の NVIDIA プラグインを使う場合に有効にする |
| **target_compile_definitions(${target} PUBLIC JETSON)** | NVIDIA Jetson でビルドする時に有効にする |
| **target_compile_definitions(${target} PUBLIC GST_APPLE)** | H.264 のデコードに GStreamer の Apple VideoToolbox プラグインを使う場合に有効にする |

### ビルド手順
```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make install

# install ディレクトリ内に実行ファイル marcon が出来る。
```

## 実行
```bash
# install ディレクトリにて、 
./marcon
```

## キーバインド

こんな感じのボタンパネル。

<img src="images/Marcon_Button.jpg" width="75%">


### 共通
| key | description |
| --- | --- |
| Enter | go to Live View |
| ESC | back to previous |
| Shift + Q | exit Application |

### Camera Main パネル
| key | description |
| --- | --- |
| W | go to FPS panel |
| E | go to ISO panel |
| R | go to Shutter panel |
| X | go to ND panel |
| C | go to IRIS panel |
| V | go to White Balance panel |
| Space | Click **REC** button |

### Camera Control パネル
| key | description |
| --- | --- |
| E, cursor UP | select item above |
| V, curos DOWN | select item below |
| X | go to MODE select panel |
| W | go to MODE(another) select panel |

## マウス/タッチ操作
### 押せそうなボタンっぽい箇所
押すとおおよそ期待どおりの動作をする。

### ESC キーの代わり
右から左に画面いっぱいにドラッグすると ESC キーと同じ挙動をする。  
**Left <-- Right**

### (Camera Control パネル) X キーの代わり
左から右に画面いっぱいにドラッグすると X キーと同じ挙動をする。  
(go to MODE select panel)  
**Left --/ Right**

ただし、
W キーが有効な時(MODE(another)がある時)は、
斜めに左下から右上に画面いっぱいにドラッグすると X キーと同じ挙動をする。  

### (Camera Control パネル) W キーの代わり
斜めに左上から右下に画面いっぱいにドラッグすると W キーと同じ挙動をする。  
(go to MODE(another) select panel)  
**Left --\\ Right**

### (Launcher パネル) アプリケーションの終了
ぐちゃぐちゃぐちゃ～っとマウスをドラッグするかタッチパネルをこすると
アプリケーションを終了する。
左右に 5 往復くらいかな。

### Camera の IP アドレスの検索
Launcher パネルの Search ボタンを押すとネットワーク内を検索する。

<img src="images/Marcon_Launcher_before_IP_search.png" width="50%">

IP アドレスが見つかると、三角形のボタンが現れる。
このボタンを押すと IP アドレスがリストアップされ選択できる。

<img src="images/Marcon_Launcher_after_IP_search.png" width="50%">

### テキスト入力
IP アドレス、ポート番号、ユーザー名、パスワードを入力する時に
テキスト入力領域をクリック/タップすると、キー入力パネルがポップアップ表示される。
入力できる文字は ASCII コード 0x00~0xFF の範囲内の英(大小)数字と記号と空白文字。

#### キーレイアウト
<img src="images/Marcon_Launcher_Input_layout.png" width="100%">

<img src="images/Marcon_Launcher_Input_0.png" width="50%">

<img src="images/Marcon_Launcher_Input_1.png" width="50%">

<img src="images/Marcon_Launcher_Input_2.png" width="50%">

### タリーランプ点滅
Launcher パネル、Camera Main パネル、Live View パネルにある **"@T@"** ボタンを押すと
タリー(赤)が約 3 秒間点滅する。

<img src="images/Marcon_Camera_Main_panel.png" width="50%">

### PTZF(パン、チルト、ズーム、フォーカス)操作
PTZF 操作パネルとタッチフォーカス操作パネルとを切り替えながら操作する。
切り替えは画面左下のボタン(PTZ, Focus)で行う。

#### PTZF 操作パネル
##### パン、チルト
画面中央付近から上下左右方向にドラッグする。  
正面に戻すには下記の操作をする。
* 画面中央付近をダブルクリック/ダブルタップ
* 画面中央付近をクリック/タップしたまま Enter キーを押す
* Enter キーを押したまま画面中央付近をクリック/タップする

##### ズーム
画面左寄りの領域を上下にドラッグする。

##### フォーカス(マニュアル)
画面右寄りの領域を上下にドラッグする。

<img src="images/Marcon_Camera_LiveView_PTZF.png" width="75%">

パン、チルトをリセットするには画面右下の **"Reset"** ボタンを押す。

<img src="images/Marcon_Camera_LiveView_PT_reset.png" width="75%">

#### タッチフォーカス操作パネル
画面のほぼ全域でタッチ操作が出来る。  
タッチ操作は下記のいずれかで OK。
* ダブルクリック/ダブルタップ
* クリック/タップしたまま Enter キーを押す
* Enter キーを押したままクリック/タップする

<img src="images/Marcon_Camera_LiveView_TouchFocus.png" width="75%">

## Ubuntu のディスプレイ表示の回転
時々、ディスプレイ表示が縦長なのがある。
それを横長にするには、
[ここの Web サイト](http://bluearth.cocolog-nifty.com/blog/2019/12/post-e5f4f1.html)
を参考すると良い。

右回転のスクリプトはこれ。
適宜、LCDのデバイス名(例、***WaveShare WS170120***)は変更してね。
```bash
#!/bin/bash

SLEEP_SEC=${1:-0}

sleep ${SLEEP_SEC}

export DISPLAY=:0.0

# display rotation.
xrandr --output HDMI-0 --rotate right

# pointing device rotation.
LCD_ID=$(xinput | grep 'WaveShare WS170120' | perl -pe 's/^.*\Wid=([0-9]+)\W.*$/${1}/')
xinput set-prop ${LCD_ID} 'Coordinate Transformation Matrix' 0 1 0 -1 0 1 0 0 1
```

## GUI のスキン変更
起動直後の表示パネルの最上部左寄りに **"Menu"** があるので、
そこの **"Change skin"** から適宜選択する。

<img src="images/Marcon_Menu.jpg" width="75%">

## テストプログラムたち
test ディレクトリ以下に置いてある。
* [cgi_test](./test/cgi_test/)
* [cgi_test_cpprestsdk](./test/cgi_test_cpprestsdk/)
* [cgi_test_cpr](./test/cgi_test_cpr/)
* [srt_receive](./test/srt_receive/)
