#date: 2022-06-06T17:05:58Z
#url: https://api.github.com/gists/2089b8be2c70e7c97533d3d92c4bdeb7
#owner: https://api.github.com/users/Lesmiscore

import tempfile
import time
import subprocess
import sys
from yt_dlp.ts_parser import (
    write_mp4_boxes,
    parse_mp4_boxes,
    pack_be32,
    pack_be64,
    unpack_be32,
    unpack_be64,
)

# with open("test/test.mp4", 'rb') as r:
#     with tempfile.TemporaryFile('w+b') as w:
#         write_mp4_boxes(w, parse_mp4_boxes(r))
#         w.seek(0)
#         r.seek(0)
#         assert w.read() == r.read()

with open("test/test.mp4", 'rb') as r:
    # proc = subprocess.Popen(['ffmpeg', '-i', '-', '-f', 'null', '/dev/null'], stdin=subprocess.PIPE)
    # proc = subprocess.Popen(['mpv', '-'], stdin=subprocess.PIPE)
    # with proc.stdin as w:
    with sys.stdout.buffer as w:
        def aaa():
            base_bmdt = -1
            for box in parse_mp4_boxes(r):
                # time.sleep(0.01)
                if box[0] == 'tfdt':
                    content = box[1]
                    version = unpack_be32(content[0:4])
                    # baseMediaDecodeTime always comes to the first
                    if version == 0:
                        bmdt = unpack_be32(content[4:8])
                    else:
                        bmdt = unpack_be64(content[4:12])
                    if bmdt == 0:
                        yield box
                        continue
                    # calculate new baseMediaDecodeTime
                    ob = bmdt
                    if base_bmdt < 0:
                        base_bmdt = bmdt
                        bmdt = 0
                    else:
                        bmdt = max(0, bmdt - base_bmdt)
                    print(f'{ob} -> {bmdt}', file=sys.stderr)
                    # pack everything again and insert as a new box
                    if version == 0:
                        bmdt_b = pack_be32(bmdt)
                    else:
                        bmdt_b = pack_be64(bmdt)
                    yield ('tfdt', content[0:4] + bmdt_b + content[8 + version * 4:])
                    continue
                yield box

        write_mp4_boxes(w, aaa())

