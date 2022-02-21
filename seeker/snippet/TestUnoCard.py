#date: 2022-02-21T16:53:57Z
#url: https://api.github.com/gists/3138e63bb2e6f3adbf77354b44c6c351
#owner: https://api.github.com/users/valentina-aparicio

# Unit-testing on UnoCard
# Valentina Aparicio, 17 Feb 2022

import unittest
from typing import List
from AbstractCard import AbstractCard
from UnoCard import UnoCard

class TestUnoCard(unittest.TestCase):
    # All methods whose names start with "test"
    # will be treated as tests
    def test_create_green_5(self) -> None:
        self.assertEqual(str(UnoCard('5', 'Green')), 'green 5')

    def test_create_all_color(self) -> None:
        for rank in UnoCard._COLOR_RANKS:
            with self.subTest(r=rank):
                for suit in UnoCard._COLOR_SUITS:
                    with self.subTest(s=suit):
                        self.assertEqual(str(UnoCard(rank, suit)),
                             suit + ' ' + rank)

    def test_create_all_wild(self) -> None:
        self.assertEqual(str(UnoCard('', 'wild')), 'wild')
        self.assertEqual(str(UnoCard('draw 4', 'wild')), 'wild draw 4')

    def test_invalid_rank(self) -> None:
        with self.assertRaises(AssertionError):
            UnoCard('', 'red')

    def test_invalid_suit(self) -> None:
        with self.assertRaises(AssertionError):
            UnoCard('3', 'kidneys')

    def testSuit(self) -> None:
        for rank in UnoCard._COLOR_RANKS:
            with self.subTest(r=rank):
                for suit in UnoCard._COLOR_SUITS:
                    with self.subTest(s=suit):
                        self.assertEqual(UnoCard(rank, suit).suit(), suit)

        for rank in UnoCard._WILD_RANKS:
            with self.subTest(r=rank):
                for suit in UnoCard._SUITS[:-1]:
                    with self.subTest(s=suit):
                        self.assertEqual(UnoCard(rank, suit).suit(), suit)

    def testRank(self) -> None:
        for rank in UnoCard._COLOR_RANKS:
            with self.subTest(r=rank):
                for suit in UnoCard._COLOR_SUITS:
                    with self.subTest(s=suit):
                        self.assertEqual(UnoCard(rank,suit).rank(),
                                        UnoCard._COLOR_RANKS.index(rank))

        for rank in UnoCard._WILD_RANKS:
            with self.subTest(r=rank):
                for suit in UnoCard._SUITS[:-1]:
                    with self.subTest(s=suit):
                        self.assertEqual(UnoCard(rank, suit).rank(),
                                        UnoCard._WILD_RANKS.index(rank))

    def testRankName(self) -> None:
        for rank in UnoCard._COLOR_RANKS:
            with self.subTest(r=rank):
                for suit in UnoCard._COLOR_SUITS:
                    with self.subTest(s=suit):
                        self.assertEqual(UnoCard(rank, suit).rankName(),
                                            rank)

        for rank in UnoCard._WILD_RANKS:
            with self.subTest(r=rank):
                for suit in UnoCard._SUITS[:-1]:
                    with self.subTest(s=suit):
                        self.assertEqual(UnoCard(rank, suit).rankName(),
                                            rank)

    def testMakeDeck(self) -> None:
        deck: List[AbstractCard] = UnoCard.makeDeck()
        self.assertEqual(len(deck), 108)

        for i in range(52):
            self.assertEqual(deck[i].suit(), UnoCard._COLOR_SUITS[i % 4])
            self.assertEqual(deck[i].rank(), (i // 4) + 1)

        for i in range(48):
            self.assertEqual(deck[i].suit(), UnoCard._COLOR_SUITS[i % 4])
            self.assertEqual(deck[i].rank(), (i // 4) + 1)

        for i in range(8):
            self.assertEqual(deck[i].suit(), UnoCard._SUITS[:-1])
            self.assertEqual(deck[i].rank(), (i + 1))
        
if __name__ == '__main__':
    unittest.main()