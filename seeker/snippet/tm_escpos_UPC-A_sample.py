#date: 2025-07-21T16:44:59Z
#url: https://api.github.com/gists/e642c04284ba823df2ad7c7ca4a7e168
#owner: https://api.github.com/users/hatotank

# UPC-Aを印刷するためのサンプルコード
# 事前にescpos-pythonライブラリをインストールしておくこと
# 詳細なコマンド仕様は以下のURLを参照
# https://support.epson.net/publist/reference_ja/
from escpos.printer import Network

p = Network("192.168.10.55")  # ← 自分のプリンタのIPに変更

# 初期化
p._raw(b'\x1b\x40')  # ESC @
p.text("== UPC-A Sample ==\n")

# GS H
# HRI文字の印字位置の選択
# n = 2 印字しない / 1 バーコードの上 / 2 バーコードの下 / 3 バーコードの上と下
p._raw(b'\x1d\x48\x02')  # GS H n

# GS f
# HRI文字のフォントの選択
# n = 0 フォントA / 1 フォントB
p._raw(b'\x1d\x66\x00')  # GS f n

# GS h
# バーコードの高さの設定
# n = 1 ～ 255
p._raw(b'\x1d\x68\x70')  # GS h n

# GS w
# バーコードの幅の設定
# n = 2 ～ 6
p._raw(b'\x1d\x77\x03')  # GS w n

# GS k
# バーコードの印字 <機能 B>
symbol_data = b'12345678901'  # 11桁の数字(チェックデジット1桁)

command = bytearray()
command.extend(b'\x1d\x6b')       # GS k
command.extend(b'\x41')           # m = 65 (UPC-A)
command.append(len(symbol_data))  # n データ数
command.extend(symbol_data)       # d1 ～ d11 UPC-Aの11桁の数字

p._raw(command)                   # GS k m n d1...dk

# 余白と用紙カット
p.text("\n\n")
p.cut()