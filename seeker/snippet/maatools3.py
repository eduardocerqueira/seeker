#date: 2026-02-26T17:40:25Z
#url: https://api.github.com/gists/b3d2dd38c8a361a2dbb4e904162bdcc9
#owner: https://api.github.com/users/hguandl

# /// script
# dependencies = [
#   "opencv-python",
#   "numpy",
# ]
# ///

import argparse
import socket
import struct

import cv2
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('output')
parser.add_argument('--host', default='localhost')
parser.add_argument('--port', default=1717, type=int)
parser.add_argument('--bgr', action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((args.host, args.port))

    with sock.makefile('rb') as f:
        sock.sendall(b'MAA\x00')
        data = f.read(4)
        assert data == b'OKAY'

        sock.sendall(b'\x00\x04VERN')
        version, = struct.unpack('>I', f.read(4))
        assert version >= 3
        print("MaaTools version:", version)

        sock.sendall(b'\x00\x04BNDL')
        bundle_length, = struct.unpack('>I', f.read(4))
        bundle_data = f.read(bundle_length)
        bundle_id = bundle_data.decode('utf-8')
        print("Bundle ID:", bundle_id)

        sock.sendall(b'\x00\x04RECT')
        t = struct.unpack('>hhhhhhhh', f.read(16))
        print("Window Rect:", (t[:2], t[2:4]))
        print("Content Rect:", (t[4:6], t[6:]))

        sock.sendall(b'\x00\x04SIZE')
        width, height = struct.unpack('>HH', f.read(4))
        print("Native size:", (width, height))

        if args.bgr:
            sock.sendall(b'\x00\x04BGR\x01')
            width, height, image_length = struct.unpack('>III', f.read(12))
            print("Image size:", (width, height))
            image = f.read(image_length)
            print("Got BGR image of bytes:", image_length)
            mat = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 3))
        else:
            sock.sendall(b'\x00\x04SCRN')
            image_length, = struct.unpack('>I', f.read(4))
            image = f.read(image_length)
            print("Got RGBA image of bytes:", image_length)
            mat = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
            mat = cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)

        cv2.imwrite(args.output, mat)
