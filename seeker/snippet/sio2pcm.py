#date: 2025-08-14T16:44:42Z
#url: https://api.github.com/gists/580d7bea36185c5a0ef759b1fcf6df74
#owner: https://api.github.com/users/rlcamp

#!/usr/bin/env python3
# reads a big-endian signed 16-bit .sio file and emits raw pcm samples
# usage: ./sio2pcm.py < J1312340.hla.south.sio > J1312340.hla.south.pcm
import sys
import struct
import numpy as np

idfield, number_of_records, record_length_bytes_per_channel, C, bytes_per_sample, f0, total_samples_per_channel, endianness_indicator = struct.unpack('>IIIIIIII', sys.stdin.buffer.read(32))

if 32677 != endianness_indicator:
    print('not a big endian sio file', file=sys.stderr)
    sys.exit()

if 2 != bytes_per_sample:
    print('not a 16-bit sio file', file=sys.stderr)
    sys.exit()

filename = sys.stdin.buffer.read(24).decode('utf-8')
comment = sys.stdin.buffer.read(72).decode('utf-8')

if number_of_records % C:
    print('warning: number of channels does not evenly divide nr', file=sys.stderr)

if record_length_bytes_per_channel % bytes_per_sample:
    print('warning: bytes per sample does not evenly divide bytes per channel', file=sys.stderr)

samples_per_record_per_channel = record_length_bytes_per_channel // bytes_per_sample

if total_samples_per_channel % samples_per_record_per_channel * number_of_records:
    print('warning: samples per record per channel does not evenly divide total samples per channel', file=sys.stderr)

# read rest of void space before first record
discard = sys.stdin.buffer.read(record_length_bytes_per_channel - 128)

for irec in range(number_of_records // C):
    # read one block of records for all channels
    record = np.fromfile(sys.stdin, dtype='>i2', count=(samples_per_record_per_channel * C))

    # interpret the above as a two dimensional array with this shape
    record.shape = (C, samples_per_record_per_channel)

    # transpose it and write it back out as host-endian int16
    sys.stdout.buffer.write(np.transpose(record).astype(np.int16).copy(order='C'))
