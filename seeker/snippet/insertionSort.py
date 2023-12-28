#date: 2023-12-28T16:53:02Z
#url: https://api.github.com/gists/e662022bbed4702d44c2cdc718f6c0e1
#owner: https://api.github.com/users/primaryobjects

from typing import List
from collections import namedtuple

Pair = namedtuple('Pair', ['key', 'value'])

# Definition for a pair.
# class Pair:
#     def __init__(self, key: int, value: str):
#         self.key = key
#         self.value = value
class Solution:
    def insertionSort(self, pairs: List[Pair]) -> List[List[Pair]]:
        history = [pairs.copy()] if pairs else []
        sorted_pairs = pairs

        for i in range(0, len(sorted_pairs) - 1):
            a = sorted_pairs[i].key
            b = sorted_pairs[i+1].key

            j = i
            while b < a and j > -1:
                # Swap positions.
                temp = sorted_pairs[j+1]
                sorted_pairs[j+1] = sorted_pairs[j]
                sorted_pairs[j] = temp

                a = sorted_pairs[j - 1].key
                b = sorted_pairs[j].key

                j = j - 1

            history.append(sorted_pairs.copy())

        return history