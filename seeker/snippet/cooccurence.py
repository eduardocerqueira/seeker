#date: 2021-10-20T17:14:18Z
#url: https://api.github.com/gists/0f0f10ccdd6ac29c175d2f12204d56cf
#owner: https://api.github.com/users/GreatBahram

from abc import ABC, abstractmethod


class CoOccurence(ABC):
    def __init__(self) -> None:
        self._container = None

    @abstractmethod
    def update(self, line: str) -> None:
        ...

    @abstractmethod
    def get(self, word1: int, word2: str) -> int:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self._container)})"


# co = CoOccurence()

# co.update("alice bob martin")
# co.update("alice claire")

# co.get("alice bob") == co.get("bob alice") == 1
# co.get("martin claire") == 0


#### FIRST TRY! #####


class CoOccurenceMap(CoOccurence):
    """Solve using simple data structure."""

    def __init__(self) -> None:
        self._container: list[set[str]] = []

    def update(self, line: str) -> None:
        self._container.append({word for word in line.split()})

    def get(self, word1: str, word2: str) -> int:
        count: int = 0
        for word_set in self._container:
            if word1 in word_set and word2 in word_set:
                count += 1
        return count

    def get2(self, word1: str, word2: str) -> int:
        """Expediate only a bit using generator expression."""
        return sum(
            word1 in word_set and word2 in word_set for word_set in self._container
        )


co1 = CoOccurenceMap()
co1.update("alice bob martin")
co1.update("alice claire")

assert co1.get("alice", "bob") == co1.get("bob", "alice") == 1
assert co1.get("martin", "claire") == 0

### Second Try ###
from collections import defaultdict


class CoOccurenceWordLineMap(CoOccurence):
    def __init__(self) -> None:
        self._container: dict[str, set[int]] = defaultdict(set)
        self._line_num: int = 0

    def update(self, line: str) -> None:
        self._line_num += 1
        for word in line.split():
            self._container[word].add(self._line_num)

    def get(self, word1: str, word2: str) -> int:
        word1_set = self._container[word1]
        word2_set = self._container[word2]
        return len(word1_set & word2_set)


co2 = CoOccurenceWordLineMap()
co2.update("alice bob martin")
co2.update("alice claire")
assert co2.get("alice", "bob") == co2.get("bob", "alice") == 1
assert co1.get("martin", "claire") == 0
print(co2)


### Third Try ####
import itertools
from collections import defaultdict


class CoOccurencePairWordMap(CoOccurence):
    def __init__(self):
        self._container: dict[str, int] = defaultdict(int)

    def update(self, line: str) -> int:
        def iter_word_pairs(line: str):
            return itertools.combinations(line.split(), 2)

        for word1, word2 in iter_word_pairs(line):
            self._container[(word1, word2)] += 1

    def get(self, word1: str, word2: str) -> int:
        return self._container[(word1, word2)] or self._container[(word2, word1)]


co3 = CoOccurencePairWordMap()
co3.update("alice bob martin")
co3.update("alice claire")
print(co3)
assert co2.get("alice", "bob") == co2.get("bob", "alice") == 1
assert co1.get("martin", "claire") == 0


#### Fourth Try ####
# NOTE: let's imagine someone gives us the list of
# unique vocabulary from the input


class CoOccurenceMatrixWord(CoOccurence):
    def __init__(self, words: list[str]) -> None:
        self._container: list[list[int]] = []
        self._word_idx: dict[str, int] = {}
        for idx, word in enumerate(words):
            self._word_idx[word] = idx
            self._container.append([0] * len(words))

    def update(self, line: str) -> int:
        def iter_word_pairs(line: str):
            return itertools.combinations(line.split(), 2)

        for word1, word2 in iter_word_pairs(line):
            word1_loc: int = self._word_idx[word1]
            word2_loc: int = self._word_idx[word2]
            self._container[word1_loc][word2_loc] += 1

    def get(self, word1: str, word2: str) -> int:
        word1_loc: int = self._word_idx[word1]
        word2_loc: int = self._word_idx[word2]
        return self._container[word1_loc][word2_loc]


co4 = CoOccurenceMatrixWord(["alice", "bob", "martin", "claire"])
co4.update("alice bob martin")
co4.update("bob martin")
co4.update("alice claire")
print(co4)
assert co2.get("alice", "bob") == co2.get("bob", "alice") == 1
assert co1.get("martin", "claire") == 0