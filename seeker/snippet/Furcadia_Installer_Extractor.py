#date: 2022-07-01T17:00:27Z
#url: https://api.github.com/gists/92981e15accdec66a719dae01cd3fbf6
#owner: https://api.github.com/users/FelixWolf

#!/usr/bin/env python3
import struct
import io
from libfurc import archive

def unpack(data):
    data.seek(-10, 2)
    if data.read(2) != b"FC":
        print("Not a furcadia installer!")
        exit(1)
    
    size, = struct.unpack("<I", data.read(4))
    data.seek(-size - 10, 2)
    data = io.BytesIO(data.read(size))
    rch = archive.RCHFile.open(data)
    for file in rch.files:
        if file.name.lower() == "lev02.ds": #"furcadia.exe":
            print(file.read())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Unpack a furcadia installer.')
    parser.add_argument('furcadia', metavar="furcadia.exe", nargs="?", default='furcadia.exe',
                        type=argparse.FileType('rb'), help='Input installer')
    parser.add_argument('output', nargs="?", default=None,
                        help='Output folder (default: furcadia_<version>)')
    
    args = parser.parse_args()
    unpack(args.furcadia)
    

