#date: 2022-01-05T17:02:20Z
#url: https://api.github.com/gists/b1d6ae3a5872970b563189fac463f721
#owner: https://api.github.com/users/jepler

from trackdata import trackdata, samplerate

MFM_SYNC = "100010010001"

t2_time = round(samplerate * 2e-6)
t2_5_time = round(samplerate * 2.5e-6)
t3_time = round(samplerate * 3e-6)
t3_5_time = round(samplerate * 3.5e-6)
t4_time = round(samplerate * 4e-6)

def decode_flux_to_bitstream(data):
    for i in data:
        yield 1
        if i > t3_5_time:
            yield 0
        if i > t2_5_time:
            yield 0
        yield 0

bitstream = decode_flux_to_bitstream(trackdata)
decoded = "".join(str(i) for i in bitstream)

#for i in range(0, len(decoded), 72):
#    print(decoded[i:i+72])

parts = decoded.split(MFM_SYNC)

for p in parts:
    print(f"{len(p):4} {p[:72]}")
    even = int(p[::2], 2)
    print(even.to_bytes(len(p) // 16 + 1, 'big'))
    print()
