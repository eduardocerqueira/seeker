#date: 2024-04-11T16:55:06Z
#url: https://api.github.com/gists/4b4994d2284fab6eaea66360f6610a57
#owner: https://api.github.com/users/Kentemie

from pprint import pprint as print
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


# < --------------------------------------------------------------------- > #


def _search_offset(game_stamps, target_offset, key=None):
    left, right = 0, len(game_stamps)
    
    while left < right:
        mid = (left + right) // 2

        if key(game_stamps[mid]) < target_offset:
            left = mid + 1
        else:
            right = mid

    return left

def get_score(game_stamps, offset):
    '''
    Takes list of game's stamps and time offset for which returns the scores for the home and away teams.
    Please pay attention to that for some offsets the game_stamps list may not contain scores.
    # ---------------------------------------------------- #
    Algorithm:
    Due to the fact that the game_stamps list is sorted by offset (I think so because of the implementaion of the
    generate_stamp function), I want to use a binary search algorithm. If there are no scores for a given offset, 
    I will return the scores for the offset in game_stamps that precedes the given one.
    '''
    if not game_stamps:
        print("The game hasn`t started yet.")
        return 0, 0
    if offset < game_stamps[0]['offset'] or offset > game_stamps[-1]['offset']:
        print("Please enter the correct time offset.")
        return 0, 0
    
    idx = _search_offset(game_stamps, offset, lambda x: x['offset'])

    home = game_stamps[idx]['score']['home']
    away = game_stamps[idx]['score']['away']

    return home, away
