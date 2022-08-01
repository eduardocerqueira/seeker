#date: 2022-08-01T17:08:59Z
#url: https://api.github.com/gists/94bc6954a9491916581ffd291addd36e
#owner: https://api.github.com/users/v3l0c1r4pt0r

#!/usr/bin/env python3
# convert DSView TXT dump of USB packet fields into valid PCAP file
import re
import sys
import struct
import os
from enum import IntEnum

class PCAP_Header():

  def __init__(self):
    self.magic = 0xa1b2c3d4
    self.version_major = 2
    self.version_minor = 4
    self.thiszone = 0
    self.sigfigs = 0
    self.snaplen = 0x40000
    self.network = 288

  def __bytes__(self):
    return struct.pack("<IHHIIII", self.magic, self.version_major, self.version_minor, self.thiszone, self.sigfigs, self.snaplen, self.network)


class PCAPrec_Header():

  def __init__(self, length):
    self.ts_sec = 0
    self.ts_usec = 0
    self.incl_len = length
    self.orig_len = length

  def __bytes__(self):
    return struct.pack("<IIII", self.ts_sec, self.ts_usec, self.incl_len, self.orig_len)


class PID(IntEnum):
  SETUP = 0x2d
  DATA1 = 0x4b
  NAK = 0x5a
  IN = 0x69
  SOF = 0xa5
  DATA0 = 0xc3
  ACK = 0xd2
  OUT = 0xe1


class SOF:

  def __init__(self, stream):
    self.pid = PID.SOF
    line = stream.readline()
    if re.match(r'.*Frame.*', line):
      frame = line.strip().split(',')[2].split(':')[1].strip()
      self.frame = int(frame)
    else:
      raise Exception('Frame not found')
    line = stream.readline()
    if re.match(r'.*CRC5.*', line):
      crc5 = line.strip().split(',')[2].split(':')[1].strip()
      self.crc5 = int(crc5, base=16)
    else:
      raise Exception('CRC5 not found')

  def __bytes__(self):
    return struct.pack("<BH", int(self.pid), self.frame | (self.crc5 << 11))


class SETUP:

  def __init__(self, stream):
    self.pid = PID.SETUP
    line = stream.readline()
    if re.match(r'.*Address.*', line):
      address = line.strip().split(',')[2].split(':')[1].strip()
      self.address = int(address)
    else:
      raise Exception('Address not found')
    line = stream.readline()
    if re.match(r'.*Endpoint.*', line):
      endpoint = line.strip().split(',')[2].split(':')[1].strip()
      self.endpoint = int(endpoint)
    else:
      raise Exception('Endpoint not found')
    line = stream.readline()
    if re.match(r'.*CRC5.*', line):
      crc5 = line.strip().split(',')[2].split(':')[1].strip()
      self.crc5 = int(crc5, base=16)
    else:
      raise Exception('CRC5 not found')

  def __bytes__(self):
    return struct.pack("<BH", int(self.pid), self.address | (self.endpoint << 7) | (self.crc5 << 11))


class IN:

  def __init__(self, stream):
    self.pid = PID.IN
    line = stream.readline()
    if re.match(r'.*Address.*', line):
      address = line.strip().split(',')[2].split(':')[1].strip()
      self.address = int(address)
    else:
      raise Exception('Address not found')
    line = stream.readline()
    if re.match(r'.*Endpoint.*', line):
      endpoint = line.strip().split(',')[2].split(':')[1].strip()
      self.endpoint = int(endpoint)
    else:
      raise Exception('Endpoint not found')
    line = stream.readline()
    if re.match(r'.*CRC5.*', line):
      crc5 = line.strip().split(',')[2].split(':')[1].strip()
      self.crc5 = int(crc5, base=16)
    else:
      raise Exception('CRC5 not found')

  def __bytes__(self):
    return struct.pack("<BH", int(self.pid), self.address | (self.endpoint << 7) | (self.crc5 << 11))


class OUT:

  def __init__(self, stream):
    self.pid = PID.OUT
    line = stream.readline()
    if re.match(r'.*Address.*', line):
      address = line.strip().split(',')[2].split(':')[1].strip()
      self.address = int(address)
    else:
      raise Exception('Address not found')
    line = stream.readline()
    if re.match(r'.*Endpoint.*', line):
      endpoint = line.strip().split(',')[2].split(':')[1].strip()
      self.endpoint = int(endpoint)
    else:
      raise Exception('Endpoint not found')
    line = stream.readline()
    if re.match(r'.*CRC5.*', line):
      crc5 = line.strip().split(',')[2].split(':')[1].strip()
      self.crc5 = int(crc5, base=16)
    else:
      raise Exception('CRC5 not found')

  def __bytes__(self):
    return struct.pack("<BH", int(self.pid), self.address | (self.endpoint << 7) | (self.crc5 << 11))


class DATA0:

  def __init__(self, stream):
    self.pid = PID.DATA0
    self.databytes = []
    line = stream.readline()
    while re.match(r'.*Databyte.*', line):
      databyte = line.strip().split(',')[2].split(':')[1].strip()
      self.databytes.append(int(databyte, base=16))
      line = stream.readline()
    if re.match(r'.*CRC16.*', line):
      crc16 = line.strip().split(',')[2].split(':')[1].strip()
      self.crc16 = int(crc16, base=16)
    else:
      raise Exception('CRC16 not found')

  def __bytes__(self):
    b = b''
    b += struct.pack("<B", int(self.pid))

    for db in self.databytes:
      b += struct.pack("<B", db)

    b += struct.pack("<H", self.crc16)

    return b


class DATA1:

  def __init__(self, stream):
    self.pid = PID.DATA1
    self.databytes = []
    line = stream.readline()
    while re.match(r'.*Databyte.*', line):
      databyte = line.strip().split(',')[2].split(':')[1].strip()
      self.databytes.append(int(databyte, base=16))
      line = stream.readline()
    if re.match(r'.*CRC16.*', line):
      crc16 = line.strip().split(',')[2].split(':')[1].strip()
      self.crc16 = int(crc16, base=16)
    else:
      raise Exception('CRC16 not found')

  def __bytes__(self):
    b = b''
    b += struct.pack("<B", int(self.pid))

    for db in self.databytes:
      b += struct.pack("<B", db)

    b += struct.pack("<H", self.crc16)

    return b


class ACK:

  def __init__(self, stream):
    self.pid = PID.ACK

  def __bytes__(self):
    return struct.pack("<B", int(self.pid))


class NAK:

  def __init__(self, stream):
    self.pid = PID.NAK

  def __bytes__(self):
    return struct.pack("<B", int(self.pid))


ignore_sof = len(sys.argv) > 1 and sys.argv[1] == '--no-sof'
hdr = PCAP_Header()
os.write(1, bytes(hdr))
line = None
while line != '':
  line = sys.stdin.readline()
  if re.match(r'.*SYNC.*', line):
    sync = line.strip().split(',')[2]
    sync_value = int(sync.split(':')[1].strip())
    sync_id = line.strip().split(',')[0]
    print(sync_id, file=sys.stderr)
    if sync_value != 1:
      continue
    line = sys.stdin.readline()
    if re.match(r'.*PID.*', line):
      pid = line.strip().split(',')[2].split(':')[1].strip()
      #try:
      if True:
        pid = PID[pid]
        if pid == PID.SOF and not ignore_sof:
          sof = SOF(sys.stdin)
          print(sof.pid, sof.frame, sof.crc5, file=sys.stderr)
          b = bytes(sof)
          hdr = PCAPrec_Header(len(b))
          os.write(1, bytes(hdr) + b)
          #sys.exit(0)
        elif pid == PID.SETUP:
          setup = SETUP(sys.stdin)
          print(setup.pid, setup.address, setup.endpoint, setup.crc5, file=sys.stderr)
          b = bytes(setup)
          hdr = PCAPrec_Header(len(b))
          os.write(1, bytes(hdr) + b)
        elif pid == PID.IN:
          inn = IN(sys.stdin)
          print(inn.pid, inn.address, inn.endpoint, inn.crc5, file=sys.stderr)
          b = bytes(inn)
          hdr = PCAPrec_Header(len(b))
          os.write(1, bytes(hdr) + b)
        elif pid == PID.OUT:
          out = OUT(sys.stdin)
          print(out.pid, out.address, out.endpoint, out.crc5, file=sys.stderr)
          b = bytes(out)
          hdr = PCAPrec_Header(len(b))
          os.write(1, bytes(hdr) + b)
        elif pid == PID.DATA0:
          data0 = DATA0(sys.stdin)
          print(data0.pid, data0.databytes, data0.crc16, file=sys.stderr)
          b = bytes(data0)
          hdr = PCAPrec_Header(len(b))
          os.write(1, bytes(hdr) + b)
        elif pid == PID.DATA1:
          data1 = DATA1(sys.stdin)
          print(data1.pid, data1.databytes, data1.crc16, file=sys.stderr)
          b = bytes(data1)
          hdr = PCAPrec_Header(len(b))
          os.write(1, bytes(hdr) + b)
        elif pid == PID.ACK:
          ack = ACK(sys.stdin)
          print(ack.pid, file=sys.stderr)
          b = bytes(ack)
          hdr = PCAPrec_Header(len(b))
          os.write(1, bytes(hdr) + b)
        elif pid == PID.NAK:
          nak = NAK(sys.stdin)
          print(nak.pid, file=sys.stderr)
          b = bytes(nak)
          hdr = PCAPrec_Header(len(b))
          os.write(1, bytes(hdr) + b)
        else:
          print(f'Ignoring {str(pid)}', file=sys.stderr)
      #except Exception as e:
      #  print(e)
      #  continue
    else:
      continue
  else:
    pass
