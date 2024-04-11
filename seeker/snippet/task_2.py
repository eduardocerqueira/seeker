#date: 2024-04-11T16:55:06Z
#url: https://api.github.com/gists/4b4994d2284fab6eaea66360f6610a57
#owner: https://api.github.com/users/Kentemie

import unittest

from task_1 import generate_game, get_score

class TestGetScoreFunction(unittest.TestCase):

    def test_game_not_started(self):
        game_stamps = []
        home_score, away_score = get_score(game_stamps, 10)
        self.assertEqual(home_score, 0)
        self.assertEqual(away_score, 0)

    def test_invalid_offset_before_game(self):
        game_stamps = generate_game()
        home_score, away_score = get_score(game_stamps, -10)
        self.assertEqual(home_score, 0)
        self.assertEqual(away_score, 0)

    def test_invalid_offset_after_game(self):
        game_stamps = generate_game()
        home_score, away_score = get_score(game_stamps, 1_000_000_000)
        self.assertEqual(home_score, 0)
        self.assertEqual(away_score, 0)

    def test_exact_offset_match(self):
        game_stamps = generate_game()
        home_score, away_score = get_score(game_stamps, 0)
        self.assertEqual(home_score, game_stamps[0]['score']['home'])
        self.assertEqual(away_score, game_stamps[0]['score']['away'])

    def test_offset_within_game(self):
        game_stamps = generate_game()
        offset = game_stamps[len(game_stamps)//2]['offset']
        home = game_stamps[len(game_stamps)//2]['score']['home']
        away = game_stamps[len(game_stamps)//2]['score']['away']

        home_score, away_score = get_score(game_stamps, offset)
        self.assertGreaterEqual(offset, game_stamps[0]['offset'])
        self.assertLessEqual(offset, game_stamps[-1]['offset'])
        self.assertGreaterEqual(home_score, 0)
        self.assertGreaterEqual(away_score, 0)
        self.assertEqual(home, home_score)
        self.assertEqual(away, away_score)

    def test_last_offset_in_game(self):
        game_stamps = generate_game()
        offset = game_stamps[-1]['offset']
        home_score, away_score = get_score(game_stamps, offset)
        self.assertEqual(home_score, game_stamps[-1]['score']['home'])
        self.assertEqual(away_score, game_stamps[-1]['score']['away'])

if __name__ == '__main__':
    unittest.main()