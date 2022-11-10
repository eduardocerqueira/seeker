#date: 2022-11-10T17:11:08Z
#url: https://api.github.com/gists/5b97e68e386b741abd09558f49fd4a35
#owner: https://api.github.com/users/Darigraye

import random
import unittest

import sys
import main


class GetScoreTests(unittest.TestCase):
    def test_negative_offset(self):
        self.assertEqual(main.get_score(main.game_stamps, -1), (None, None))

        random_negative_offset = random.randint(-100000, -2)
        self.assertEqual(main.get_score(main.game_stamps,
                                        random_negative_offset), (None, None))

    def test_exceeding_value_offset(self):
        # max result offset 3 * 50000 = 150000,
        # because max step=3
        random_exceeding_offset = random.randint(150000, sys.maxsize)
        self.assertEqual(main.get_score(main.game_stamps,
                                        random_exceeding_offset), (None, None))

    def __attribute_is_change(self, index, attribute):
        return main.game_stamps[index]["score"][attribute] != \
               main.game_stamps[index - 1]["score"][attribute]

    def __check_changing_of_score(self, index):
        return self.__attribute_is_change(index, attribute="home") or \
               self.__attribute_is_change(index, attribute="away")

    def __there_are_missed_offsets(self, index):
        return main.game_stamps[index]["offset"] != \
               main.game_stamps[index - 1]["offset"] + 1

    def __get_offsets(self):
        change_score_offsets = {}
        # list offsets which not contained
        # in game_stamps dictionary but after
        # which the score was changed
        change_score_offsets_missed = []

        for index in range(1, len(main.game_stamps)):
            if self.__check_changing_of_score(index):
                home = main.game_stamps[index]["score"]["home"]
                away = main.game_stamps[index]["score"]["away"]
                change_score_offsets.update(
                    {main.game_stamps[index]["offset"]: (home, away)})

                if self.__there_are_missed_offsets(index):
                    change_score_offsets_missed.append(
                        main.game_stamps[index]["offset"] - 1)

        return change_score_offsets, change_score_offsets_missed

    def test_get_correct_scores(self):
        change_score_offsets, change_score_offsets_missed = self.__get_offsets()
        random_index_stamp = random.randint(0, len(main.game_stamps) - 1)

        for index in range(len(main.game_stamps)):
            # check score after change it
            if main.game_stamps[index]["offset"] in change_score_offsets:
                off = main.game_stamps[index]["offset"]
                self.assertEqual(main.get_score(main.game_stamps, off),
                                 change_score_offsets[off])

                # check score before change it
                if off - 1 in change_score_offsets_missed:
                    home = main.game_stamps[index - 1]["score"]["home"]
                    away = main.game_stamps[index - 1]["score"]["away"]
                    self.assertEqual(main.get_score(main.game_stamps, off - 1),
                                     (home, away))
        # check random case
        random_stamp = main.game_stamps[random_index_stamp]
        rand_home = random_stamp["score"]["home"]
        rand_away = random_stamp["score"]["away"]
        self.assertEqual(main.get_score(main.game_stamps, random_stamp["offset"]),
                         (rand_home, rand_away))


if __name__ == '__main__':
    unittest.main()