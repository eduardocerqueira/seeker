#date: 2022-11-10T17:10:17Z
#url: https://api.github.com/gists/682258bf3baa4a8a2dbd917ab9492ffd
#owner: https://api.github.com/users/Darigraye

from pprint import pprint
import random
import math

TIMESTAMPS_COUNT = 50000

PROBABILITY_SCORE_CHANGED = 0.0001

PROBABILITY_HOME_SCORE = 0.45

OFFSET_MAX_STEP = 3

INITIAL_STAMP = {
    "offset": 0,
    "score": {
        "home": 0,
        "away": 0
    }
}


def generate_stamp(previous_value):
    score_changed = random.random() > 1 - PROBABILITY_SCORE_CHANGED
    home_score_change = 1 if score_changed and random.random() > 1 - \
                             PROBABILITY_HOME_SCORE else 0
    away_score_change = 1 if score_changed and not home_score_change else 0
    offset_change = math.floor(random.random() * OFFSET_MAX_STEP) + 1

    return {
        "offset": previous_value["offset"] + offset_change,
        "score": {
            "home": previous_value["score"]["home"] + home_score_change,
            "away": previous_value["score"]["away"] + away_score_change
        }
    }


def generate_game():
    stamps = [INITIAL_STAMP, ]
    current_stamp = INITIAL_STAMP
    for _ in range(TIMESTAMPS_COUNT):
        current_stamp = generate_stamp(current_stamp)
        stamps.append(current_stamp)

    return stamps


game_stamps = generate_game()
pprint(game_stamps)


def _get_scores_from_game_stamps(game_stamps, index):
    return game_stamps[index]["score"]["home"], game_stamps[index]["score"]["away"]


def get_score(game_stamps, offset):
    '''
        returns the scores for the home and away teams
        if offset is incorrect (< 0 or > len(game_stamps))
        will return tuple containing None
    '''
    home = None
    away = None
    if offset >= 0:
        for index in range(len(game_stamps)):
            if game_stamps[index]["offset"] == offset:
                home, away = _get_scores_from_game_stamps(game_stamps, index)
                break
            elif game_stamps[index]["offset"] > offset:
                home, away = _get_scores_from_game_stamps(game_stamps, index - 1)
                break

    return home, away


def main():
    res = get_score(game_stamps, 67770)
    print(res)


if __name__ == "__main__":
    main()