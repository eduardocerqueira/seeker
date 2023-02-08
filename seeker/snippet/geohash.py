#date: 2023-02-08T17:08:59Z
#url: https://api.github.com/gists/e64e950a52bca967daf341315ed134a9
#owner: https://api.github.com/users/kataoka271

import numba
import numpy as np
import pandas as pd

base = "0123456789bcdefghjkmnpqrstuvwxyz"


@numba.njit
def _encode(lat, lon, nbits):
    lat_max = 90
    lat_min = -90
    lon_max = 180
    lon_min = -180
    value = 0
    i = 0
    code = []

    # longitude
    if nbits % 2 == 1:
        lon_mid = (lon_max + lon_min) / 2
        if lon_mid <= lon:
            value = (value << 1) | 1
            lon_min = lon_mid
        else:
            value = value << 1
            lon_max = lon_mid
        i += 1
    # latitude, longitude
    while i < nbits:
        lat_mid = (lat_max + lat_min) / 2
        if lat_mid <= lat:
            value = (value << 1) | 1
            lat_min = lat_mid
        else:
            value = value << 1
            lat_max = lat_mid
        i += 1
        if i % 5 == 0:
            code.append(base[value & 0x1F])
        lon_mid = (lon_max + lon_min) / 2
        if lon_mid <= lon:
            value = (value << 1) | 1
            lon_min = lon_mid
        else:
            value = value << 1
            lon_max = lon_mid
        i += 1
        if i % 5 == 0:
            code.append(base[value & 0x1F])

    return (value, ''.join(code))


@numba.njit
def encode_array(array: np.ndarray, precision: int):
    values = np.empty(array.shape[0], dtype="i8")
    codes = np.empty(array.shape[0], dtype="U10")
    for i in range(array.shape[0]):
        values[i], codes[i] = _encode(array[i, 0], array[i, 1], precision * 5)
    return (values, codes)


def test():
    geohashes = encode_array(np.array([[42.605, -5.603], [30, 40]]), 5)
    print(geohashes)
    print(pd.DataFrame.from_dict({"values": geohashes[0], "codes": geohashes[1]}))


if __name__ == "__main__":
    test()

