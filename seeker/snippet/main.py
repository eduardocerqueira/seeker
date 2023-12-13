#date: 2023-12-13T17:00:42Z
#url: https://api.github.com/gists/723dca2091fbafd2f9b48fe67c5b5c1e
#owner: https://api.github.com/users/MORIMOTO520212

import serial
import sys
import subprocess
import asyncio
from time import sleep

# 72個の信号を送る必要がある
# すなわち1byte(8bit)で表す必要がある


def get_lines(cmd):
    proc = subprocess.Popen(
        cmd, shell=False, bufsize=0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        line = proc.stdout.readline()  # byte type
        sys.stdout.flush()
        if line:
            yield line

        if not line and proc.poll() is not None:
            break

# ノーツ情報の取得


def get_notes(output) -> dict:
    out_splite = str(output).split()
    return {
        'key': int(out_splite[2]),
        'voice': int(out_splite[4])}


# フレット情報の取得
def get_flet(notes) -> int:
    key_list = [None, 64, 59, 55, 50, 45, 40]  # 1弦から6弦までの先頭のkey情報
    return notes['key'] - key_list[notes['voice']]

# ビット変換


def get_bit(onoff, notes):
    send_data_bit = (onoff << 7) + \
        (notes['voice']-1 << 4) + get_flet(notes)
    return send_data_bit

# ピッキング


def picking(send_data_bit):
    sleep(0.3)
    send_data_bit |= 0b00001111
    ser.write(bytes([send_data_bit]))
    print(f"picking: bit: {bin(send_data_bit)}")


def main(path_tuxguitar, ser):

    buff = ""
    for line in get_lines(path_tuxguitar):
        if buff != str(line):
            # ノーツ送信
            if "sendNote" in str(line):
                notes = get_notes(line)
                if notes['voice'] < 0:
                    continue

            if "sendNoteOn" in str(line):
                send_data_bit = get_bit(1, notes)
                ser.write(bytes([send_data_bit]))
                print(
                    f"sendNoteOn:  bit: {bin(send_data_bit)} string: {notes['voice']} flet: {get_flet(notes)}")

                # ピッキング送信
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, picking, send_data_bit)

            elif "sendNoteOff" in str(line):
                send_data_bit = get_bit(0, notes)
                ser.write(bytes([send_data_bit]))
                print(
                    f"sendNoteOff: bit: {bin(send_data_bit)} string: {notes['voice']} flet: {get_flet(notes)}")

            else:
                pass

            buff = str(line)


if __name__ == '__main__':
    path_tuxguitar = "C:\\Program Files (x86)\\tuxguitar-1.5.6\\tuxguitar.exe"
    try:
        ser = serial.Serial('COM6', 9600, timeout=None)  # pySerial初期化
    except:
        print("ポート番号にアクセスできません。")
    asyncio.run(main(path_tuxguitar, ser))