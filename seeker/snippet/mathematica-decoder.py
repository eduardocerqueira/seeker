#date: 2022-09-02T17:09:28Z
#url: https://api.github.com/gists/c44a7d51e55236fb56ba3fea4f8f4f8f
#owner: https://api.github.com/users/egormkn

#!/usr/bin/env python3

from argparse import ArgumentParser

huffman_table = [
    "1111110101100100000010",
    "1111110101100100000011",
    "1111110101100100000100",
    "1111110101100100000101",
    "1111110101100100000110",
    "1111110101100100000111",
    "1111110101100100001000",
    "1111110101100100001001",
    "1111110101100100001010",
    "00111",
    "10101",
    "1111110101100100001011",
    "1111110101100100001100",
    "1111110101100100001101",
    "1111110101100100001110",
    "1111110101100100001111",
    "1111110101100100010000",
    "1111110101100100010001",
    "1111110101100100010010",
    "1111110101100100010011",
    "1111110101100100010100",
    "1111110101100100010101",
    "1111110101100100010110",
    "1111110101100100010111",
    "1111110101100100011000",
    "1111110101100100011001",
    "1111110101100100011010",
    "1111110101100100011011",
    "1111110101100100011100",
    "1111110101100100011101",
    "1111110101100100011110",
    "1111110101100100011111",
    "110",
    "01100000101",
    "0000011",
    "10110110111",
    "101101101100",
    "1111110101101",
    "1111110100",
    "11111101011000",
    "0100100",
    "0100101",
    "11101100",
    "10111110",
    "11100",
    "1111011",
    "1110111",
    "1001111",
    "101100",
    "111100",
    "010110",
    "1011110",
    "1011010",
    "0101110",
    "0000010",
    "0110001",
    "0101111",
    "0100001",
    "1110101",
    "10111111",
    "101101101101",
    "010101",
    "01111100",
    "111111010100",
    "0111111100100",
    "010011101",
    "111011010",
    "0100110",
    "101101111",
    "011000010",
    "100111010",
    "011000000",
    "1001110110",
    "01111101",
    "011000001001",
    "111111010111",
    "101101100",
    "111111011",
    "101101110",
    "0111111101",
    "11111100",
    "0111111111",
    "01111110",
    "0100000",
    "01001111",
    "011111110011",
    "01111111000",
    "1001110111",
    "11111101011001001",
    "011000001000",
    "1111110101100101",
    "00101",
    "111111010101",
    "00110",
    "10011100",
    "1110100",
    "0110000011",
    "10001",
    "0101000",
    "011001",
    "1111010",
    "0001",
    "1001101",
    "1111101",
    "000010",
    "10100",
    "010011100",
    "111011011",
    "101110",
    "010001",
    "10010",
    "01101",
    "011110",
    "1011011010",
    "01110",
    "00100",
    "10000",
    "000011",
    "0101001",
    "011000011",
    "1111100",
    "1001100",
    "0111111110",
    "1111111",
    "0111111100101",
    "000000",
    "111111010110011",
    "111111010110010000000",
]


class HuffmanTree:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def get(self, key, offset=0):
        if offset > len(key):
            raise Exception("offset > len(key)")
        elif offset == len(key):
            return self.value, offset
        elif self.value is not None:
            return self.value, offset
        elif key[offset] == "0" and self.left is not None:
            return self.left.get(key, offset + 1)
        elif key[offset] == "1" and self.right is not None:
            return self.right.get(key, offset + 1)
        else:
            return None, offset

    def set(self, key, value, offset=0):
        if offset > len(key):
            raise Exception("offset > len(key)")
        elif offset == len(key):
            self.value = value
        elif self.value is not None:
            raise Exception(f"Overlapping keys: {key[:offset]} and {key}")
        elif key[offset] == "0":
            if self.left is None:
                self.left = HuffmanTree()
            self.left.set(key, value, offset + 1)
        elif key[offset] == "1":
            if self.right is None:
                self.right = HuffmanTree()
            self.right.set(key, value, offset + 1)
        else:
            raise AssertionError()

    @classmethod
    def build(cls, table):
        root = cls()
        for i, bits in enumerate(table):
            root.set(bits, chr(i))
        return root

    def __repr__(self):
        result = "{"
        if self.left is not None:
            result += "0: " + repr(self.left)
        if self.left is not None and self.right is not None:
            result += ", "
        elif self.value is not None:
            result += repr(self.value)
        if self.right is not None:
            result += "1: " + repr(self.right)
        result += "}"
        return result


tree = HuffmanTree.build(huffman_table)


def split_string(s: str, n: int):
    return [s[i:i + n] for i in range(0, len(s), n)]


def huffman_encode(body: str):
    result = []
    for i in range(0, len(body), 1):
        result.append(huffman_table[ord(body[i])])
    bits = "".join(result)
    # add trailing zeros to match a multiple 13 bits
    nzeros = 13 - (len(bits) % 13)
    nzeros = 0 if nzeros == 13 else nzeros
    for i in range(nzeros):
        bits += "0"
    return bits


def base95_encode(bits: str):
    result = []
    for b in split_string(bits, 13):
        # convert string of 13 bits to be in reversed order
        number = int(b[::-1], 2)
        hi = number // 95
        lo = number % 95
        c1 = chr(hi + 32)
        c2 = chr(lo + 32)
        result.append(c1)
        result.append(c2)
    return "".join(result)


def huffman_decode(bits: str):
    result = []
    value, offset = tree.get(bits)
    while value is not None:
        result += value
        value, offset = tree.get(bits, offset)
    return "".join(result)


def base95_decode(code: str):
    result = []
    for c1, c2 in split_string(code, 2):
        hi = ord(c1) - 32
        lo = ord(c2) - 32
        number = hi * 95 + lo
        b = format(number, "013b")[::-1]
        result.append(b)
    return "".join(result)


def main():
    parser = ArgumentParser(description="Encode/decode Mathematica files")
    parser.add_argument("action", choices=["encode", "decode"])
    parser.add_argument("input_file", help="Input file")
    parser.add_argument("output_file", help="Output file")
    args = parser.parse_args()

    with open(args.input_file, "r") as input_file:
        content = input_file.read()

    header = "(*!1N!*)mcm"

    if args.action == "encode":
        bits = huffman_encode(content)
        code = base95_encode(bits)

        assert huffman_decode(bits) == content
        assert base95_decode(code) == bits

        lines = [header, *split_string(code, 70)]
        result = "\n".join(lines)
    else:
        lines = content.splitlines(keepends=False)
        if not lines or lines[0] != header:
            raise ValueError("Expected header: " + header)
        code = "".join(lines[1:])
        bits = base95_decode(code)

        result = huffman_decode(bits)

    with open(args.output_file, "w") as output_file:
        output_file.write(result)


if __name__ == "__main__":
    main()
