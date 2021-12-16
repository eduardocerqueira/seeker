#date: 2021-12-16T17:04:11Z
#url: https://api.github.com/gists/16d0ca712266ef3abd35357cb2fe0768
#owner: https://api.github.com/users/unbibium

#!/usr/bin/env python3

import sys, os,math

from collections import deque

class Bitstream:
    def __init__(self, hexstring, length=math.inf):
        # length only controls when EOF flag is set
        self.chars = deque(hexstring)
        self.bits = []
        self.eof = False
        self.length = length

    def read_bit(self):
        if self.eof:
            return 0
        self.length -= 1
        if self.length == 0 or not (self.chars or any(self.bits)):
            self.eof=True
        if not self.bits:
            if self.chars:
                hexchr = self.chars.popleft()
                hexint = int(hexchr,16)
            else: # consider raising an error instead
                raise EOFError
            self.bits = deque( map(hexint.__and__, [8,4,2,1]) )
        return 1 if self.bits.popleft() else 0

    def read_hex(self, bitcount):
        whole, part = divmod(bitcount,4)
        result = ""
        for i in range(whole):
            result += "0123456789ABCDEF"[self.read_int(4)]
        if part:
            result += "0123456789ABCDEF"[self.read_int(part) << 4-part]
        return result

    def read_int(self, bitcount):
        result = 0
        for i in range(bitcount):
            result = result*2 + self.read_bit()
        return result


class Packet:
    def __init__(self, source):
        if type(source) is str: source = Bitstream(source)
        self.source = source
        self.version = source.read_int(3)
        self.type_id = source.read_int(3)
        if self.type_id == 4: # literal
            more, self.literal_int = 1, 0
            while more:
                more = source.read_bit()
                self.literal_int = self.literal_int * 16 + source.read_int(4)
            self.version_sum = self.version
            self.subpackets = None
        else:
            self.subpackets = []
            self.length_type_id = source.read_int(1)
            if self.length_type_id == 0:
                bits_in_subpacket = source.read_int(15)
                # very inefficient converting back and forth but it's OK
                hexsource = source.read_hex(bits_in_subpacket)
                stream = Bitstream(hexsource, bits_in_subpacket)
                while not stream.eof:
                    self.subpackets.append( Packet(stream) )
            else:
                subpacket_count = source.read_int(11)
                self.subpackets = list(Packet(source) for i in range(subpacket_count))
            self.version_sum = self.version + sum(sub.version_sum for sub in self.subpackets)

    def __getitem__(self, i):
        return self.subpackets[i]

    def __iter__(self, i):
        return iter(self.subpackets)

    funcs = [
        sum,
        math.prod,
        min,
        max,
        int, #unused
        lambda a: int.__gt__(*a),
        lambda a: int.__lt__(*a),
        lambda a: int.__eq__(*a)
        ]

    fmts = [
        lambda a: "(" + ("+".join(a)) + ")",
        lambda a: "(" + ("*".join(a)) + ")",
        lambda a: "min([" + (",".join(a)) + "])",
        lambda a: "max([" + (",".join(a)) + "])",
        str,
        lambda a: "(%s > %s)" % tuple(a),
        lambda a: "(%s < %s)" % tuple(a),
        lambda a: "(%s == %s)" % tuple(a)
        ]

    def __str__(self):
        if self.type_id == 4:
            return str(self.literal_int)
        fmt = Packet.fmts[self.type_id]
        return fmt(str(packet) for packet in self.subpackets)
    
    def calc(self):
        if self.type_id == 4:
            return self.literal_int
        func = Packet.funcs[self.type_id]
        return func(packet.calc() for packet in self.subpackets)
        

def part1(lines):
    bs = Bitstream(lines[0].rstrip())
    result = 0
    while not bs.eof:
        result += Packet(bs).version_sum
    return result

def part2(lines):
    bs = Bitstream(lines[0].rstrip())
    p = Packet(bs)
    print(str(p))
    result = p.calc()
    print("leftover:", bs.read_hex(32))
    return result

if __name__ == '__main__':
    if len(sys.argv)<2:
        print("Usage",sys.argv[0],"filename")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        lines = f.readlines()

    print("part1", part1(lines))

    print("part2", part2(lines))

