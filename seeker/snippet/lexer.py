#date: 2023-10-26T16:44:02Z
#url: https://api.github.com/gists/f89e299388ab40fd048616fde5760992
#owner: https://api.github.com/users/shafu0x

from enum import Enum

TEXT = "Hello my name is Sharif. How are you? Wie geht es dir?"

 "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********"T "**********"y "**********"p "**********"e "**********"( "**********"E "**********"n "**********"u "**********"m "**********") "**********": "**********"
    WORD = 1
    PUNCTUATION = 2
    QUESTION = 3
    WHITE_SPACE = 4

class Lexer:
    def __init__(self, text):
        self.text = text
        self.index = 0

    def __iter__(self): return self

    def __next__(self):
        if self.index >= len(self.text):
            raise StopIteration
        else:
            return self.tokenize()

    def parse_word(self):
        word = self.text[self.index]
        while self.peek().isalpha():
            word += self.next()
        return word

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********": "**********"
        curr_char = self.text[self.index]

        if curr_char.isalpha():
            token = "**********"
        elif curr_char == "?":
            token = "**********"
        elif curr_char == ".":
            token = "**********"
        elif curr_char == " ":
            token = "**********"
        else:
            raise Exception("Unknown token")

        self.index += 1
        return token

    def next(self):
        self.index += 1
        return self.text[self.index]

    def peek(self):
        return self.text[self.index+1]

 "**********"d "**********"e "**********"f "**********"  "**********"p "**********"r "**********"i "**********"n "**********"t "**********"t "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********": "**********"
    for token in tokens: "**********"

printt(list(Lexer(TEXT)))