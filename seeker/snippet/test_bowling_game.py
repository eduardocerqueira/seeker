#date: 2022-07-12T16:55:27Z
#url: https://api.github.com/gists/95b3e3e92ac4b88a80d17e4805a5b679
#owner: https://api.github.com/users/enderbob364

# -*- coding: utf-8 -*- #
import unittest

from bowling_game import Game


class GameTest(unittest.TestCase):

    def setUp(self):
        self.game = Game()

    def test_gutter_game(self):
        self._roll_many(0, 20)
        self.assertEqual(0, self.game.total_score())

    def test_all_ones(self):
        self._roll_many(1, 20)
        self.assertEqual(20, self.game.total_score())

    def test_one_spare(self):
        self._roll_spare()
        self.game.roll(3)
        self._roll_many(0, 17)
        self.assertEqual(16, self.game.total_score())

    def test_one_strike(self):
        self.game.roll(10)
        self.game.roll(3)
        self.game.roll(4)
        self._roll_many(0, 16)
        self.assertEqual(24, self.game.total_score())

    def test_perfect_game(self):
        self._roll_many(10, 12)
        self.assertEqual(300, self.game.total_score())

    def test_simple_game(self):
        for pins in [1, 4, 4, 5, 6, 4, 5, 5,
                     10, 0, 1, 7, 3, 6, 4, 10, 2, 8, 6]:
            self.game.roll(pins)
        self.assertEqual(133, self.game.total_score())

    def _roll_many(self, pins, num):
        for i in range(num):
            self.game.roll(pins)

    def _roll_spare(self):
        self.game.roll(5)
        self.game.roll(5)
