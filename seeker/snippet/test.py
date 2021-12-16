#date: 2021-12-16T17:04:11Z
#url: https://api.github.com/gists/16d0ca712266ef3abd35357cb2fe0768
#owner: https://api.github.com/users/unbibium

#!/usr/bin/env python3

import unittest

from bits import *

class BitstreamTest(unittest.TestCase):
    def test_odd_stuff(self):
        b = Bitstream("D2FE28")
        self.assertEqual(b.read_int(3),6)
        self.assertEqual(b.read_int(3),4)
    def test_popstr(self):
        b = Bitstream("D2FE28")
        self.assertEqual(b.read_hex(21), "D2FE28")
        b = Bitstream("D2FE28")
        self.assertEqual(b.read_hex(20), "D2FE2")

class Part1Test(unittest.TestCase):
    def test_foo(self):
        p = Packet("D2FE28")
        self.assertEqual(p.version,6)
        self.assertEqual(p.type_id,4)
        self.assertEqual(p.literal_int, 2021)
    def test_example_two(self):
        p = Packet("38006F45291200")
        self.assertEqual(p.version,1)
        self.assertEqual(p.type_id,6)
        self.assertEqual(p[0].type_id, 4)
        self.assertEqual(p[0].literal_int, 10)
        self.assertEqual(p[1].type_id, 4)
        self.assertEqual(p[1].literal_int, 20)

    def test_example_three(self):
        p = Packet("EE00D40C823060")
        self.assertEqual(p.version,7)
        self.assertEqual(p.type_id,3)
        self.assertEqual(len(p.subpackets), 3)
        self.assertEqual(p[0].type_id, 4)
        self.assertEqual(p[0].literal_int, 1)
        self.assertEqual(p[0].calc(), 1)
        self.assertEqual(p[1].type_id, 4)
        self.assertEqual(p[1].literal_int, 2)
        self.assertEqual(p[1].calc(), 2)
        self.assertEqual(p[2].type_id, 4)
        self.assertEqual(p[2].literal_int, 3)
        self.assertEqual(p[2].calc(), 3)

    def test_example_four_a(self):
        p = Packet("8A004A801A8002F478")
        self.assertEqual(p.version,4)
        self.assertEqual(p[0].version,1)
        self.assertEqual(p[0][0].version,5)
        self.assertEqual(p[0][0][0].version,6)

    def test_version_sums(self):
        p = Packet("8A004A801A8002F478")
        self.assertEqual(p.version_sum, 16)
        p = Packet("620080001611562C8802118E34")
        self.assertEqual(p.version_sum, 12)
        p = Packet("C0015000016115A2E0802F182340")
        self.assertEqual(p.version_sum, 23)
        p = Packet("A0016C880162017C3686B18A3D4780")
        self.assertEqual(p.version_sum, 31)

class TestPart2(unittest.TestCase):
    def assertCalcEquals(self, source, expected):
        self.assertEqual(Packet(source).calc(), expected, source)

    def test_list(self):
        self.assertCalcEquals("C200B40A82",3)
    def test_product(self):
        self.assertCalcEquals("04005AC33890",54)
    def test_min(self):
        self.assertCalcEquals("880086C3E88112",7)
    def test_max(self):
        self.assertCalcEquals("CE00C43D881120",9)
    def test_lt(self):                    
        self.assertCalcEquals("D8005AC2A8F0",1)
    def test_gt(self):                    
        self.assertCalcEquals("F600BC2D8F",0)
    def test_eq(self):                    
        self.assertCalcEquals("9C005AC2F8F0",0)
    def test_sums_eq(self):                    
        self.assertCalcEquals("9C0141080250320F1802104A08",1)

if __name__ == '__main__':
    unittest.main()

