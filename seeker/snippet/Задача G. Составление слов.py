#date: 2023-01-05T16:58:23Z
#url: https://api.github.com/gists/5d4c0d5a80e13d5da9f294e706c937ef
#owner: https://api.github.com/users/AspirantDrago

from uuid import uuid4

FILENAME = 'sub-words.windows.txt'
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


class Word:
    _root = None

    def __init__(self, word: str):
        self._word = word
        self._mask = set(word)
        self.children = []
        self.parents = []
        self._id = ''
        if word == '':
            Word._root = self

    def find_parents(self, word: 'Word', id: str):
        if id == self._id:
            return
        self._id = id
        if self >= word:
            yield self
            return
        for parent in self.parents:
            yield from parent.find_parents(word, id)

    def find_children(self, id: str):
        if id == self._id:
            return
        if self._word != '':
            yield self
        self._id = id
        for child in self.children:
            yield from child.find_children(id)

    def append(self, new_word: 'Word', id: str):
        if id == self._id:
            return
        self._id = id
        no_parents = True
        for parent in self.parents:
            if parent <= new_word:
                no_parents = False
                parent.append(new_word, id)
        if no_parents:
            new_word.link(self)

    def __str__(self):
        return self._word

    def __repr__(self):
        return self._word

    def link(self, other: 'Word'):
        self.children.append(other)
        other.parents.append(self)

    def __eq__(self, other):
        return self._mask == other._mask

    def __lt__(self, other):
        return self._mask < other._mask

    def __gt__(self, other):
        return self._mask > other._mask

    def __le__(self, other):
        return self._mask <= other._mask

    def __ge__(self, other):
        return self._mask >= other._mask




with open(FILENAME, 'r', encoding='utf-8') as f:
    words = f.read().splitlines()

n = int(words[0])
words = words[1:]
tree = Word('')
dc = dict()
for i, word in enumerate(words):
    new_word = Word(word)
    for parent in tree.find_parents(new_word, uuid4()):
        parent.link(new_word)
        # print(parent, new_word)
    tree.append(new_word, uuid4())
    dc[word] = new_word
    if i % 1000 == 0:
        print(round(100 * i / n, 2), '%')

k = int(input())
for _ in range(k):
    word = input()
    for child in dc[word].find_children(uuid4()):
        print(child)
