#date: 2022-07-07T17:04:48Z
#url: https://api.github.com/gists/d743ca58fe8882e7851c2c87d0ea0067
#owner: https://api.github.com/users/dwd

# Copyright 2022 Dave Cridland
# Licensed under MIT

import sys

lat, long = [float(n) for n in sys.argv[1:3]]

LIMIT = 2 ** 20
SHIFT = 6

def encode(lat: float, long: float) -> str:
    """
    Encode deciimal degree based lat/long coordinates into a Base32 string.
    :param lat:
    :param long:
    :return:
    """
    # Convert to absolute coords instead of +/-
    if long < 0:
        long += 360
    if lat < 0:
        lat += 360
    # Convert to an integral scale
    lat_int = int(lat*LIMIT)
    long_int = int(long*LIMIT)
    # Now interleave bits of each coord
    combined = 0
    for i in range(32):
        lat_bit = lat_int & (1 << i)
        long_bit = long_int & (1 << i)
        combined |= (lat_bit << i) | (long_bit << (i + 1))
    # Base32 the result
    from base64 import b32encode
    from struct import pack
    b32 = b32encode(pack('!Q', combined << SHIFT))
    return b32[0:-3].decode('ascii')


def decode(code: str) -> tuple(float, float):
    """
    Reverse the encoding.
    :param code:
    :return:
    """
    while len(code) < 13:
        code += 'A'
    code = code + '==='
    from base64 import b32decode
    from struct import unpack
    combined = unpack('!Q', b32decode(code))[0] >> SHIFT
    lat_int = 0
    long_int = 0
    for i in range(32):
        lat_int |= (combined & (1 << (2*i))) >> i
        long_int |= (combined & (1 << ((2*i) + 1))) >> (i + 1)
    lat, long = float(lat_int) / LIMIT, float(long_int) / LIMIT
    if lat > 180:
        lat -= 360
    if long > 180:
        long -= 360
    return lat, long

def display(base32_enc: str) -> str:
    """
    Convert a base32 encoded string into NATO phonetic words
    :param base32_enc:
    :return:
    """
    nato_alphabet = ['Alfa', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf', 'Hotel', 'India', 'Juliett',
                     'Kilo', 'Lima', 'Mike', 'November', 'Oscar', 'Papa', 'Quebec', 'Romeo', 'Sierra', 'Tango', 'Uniform',
                     'Victor', 'Whiskey', 'X-Ray', 'Yankee', 'Zulu']
    numbers = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Niner']
    disp = []
    for c in base32_enc:
        if ord('0') <= ord(c) <= ord('9'):
            disp.append(numbers[ord(c) - ord('0')])
        if ord('A') <= ord(c) <= ord('Z'):
            disp.append(nato_alphabet[ord(c) - ord('A')])
    return ' '.join(disp)

enc = encode(lat, long)

print(enc, len(enc), display(enc))
print(encode(int(lat), int(long)))
print(encode(-1, -1))

print(repr(decode(enc)))
print(repr((lat, long)))

def dist(coords1, coords2):
    """
    Given two decimal degree lat/long coordinates, find the distance.
    :param coords1:
    :param coords2:
    :return:
    """
    import math
    delta_lat = math.radians(coords2[0] - coords1[0]) / 2
    delta_long = math.radians(coords2[1] - coords1[1]) / 2
    lat1 = math.radians(coords1[0])
    lat2 = math.radians(coords2[0])
    x = math.sin(delta_lat) ** 2 + (math.sin(delta_long) ** 2) * math.cos(lat1) * math.cos(lat2)
    distance = math.atan2(math.sqrt(x), math.sqrt(1-x))
    return 6371000 * distance

test = ''
for c in enc:
    test += c
    coords = decode(test)
    print(enc, test, dist((lat, long), coords), repr(coords))

