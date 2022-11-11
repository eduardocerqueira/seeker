#date: 2022-11-11T16:59:28Z
#url: https://api.github.com/gists/466507a7c281f519998b00952042a976
#owner: https://api.github.com/users/shamilbi

import unittest
from task1 import get_score, INITIAL_STAMP, generate_game, ScoreException


def extract_scores(d: dict):
    return ((d2 := d['score'])['home'], d2['away'])


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.game_stamps = generate_game()

    def test_bad_offset_1(self):
        assert get_score(self.game_stamps, -1) == extract_scores(INITIAL_STAMP)

    def test_bad_offset_2(self):
        d = self.game_stamps[-1]
        offset = d['offset']
        assert get_score(self.game_stamps, offset + 1) == extract_scores(d)

    def test_bad_offset_3(self):
        with self.assertRaises(ScoreException):
            get_score(self.game_stamps, '05')

    def test_bad_list_1(self):
        assert get_score(None, 1) == extract_scores(INITIAL_STAMP)

    def test_bad_list_2(self):
        with self.assertRaises(ScoreException):
            get_score('list', 5)

    def test_bad_list_3(self):
        with self.assertRaises(ScoreException):
            get_score([{1: 2}], 5)

    def test_offset_last(self):
        d = self.game_stamps[-1]
        offset = d['offset']
        assert get_score(self.game_stamps, offset) == extract_scores(d)

    def test_offset_mid(self):
        i = len(self.game_stamps) // 2
        d = self.game_stamps[i]
        offset = d['offset']
        assert get_score(self.game_stamps, offset) == extract_scores(d)
